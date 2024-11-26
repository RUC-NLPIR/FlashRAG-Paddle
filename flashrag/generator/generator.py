from copy import deepcopy
from typing import List

import paddle
from paddle.distributed import fleet
from paddlenlp.generation import GenerationConfig
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
)
from tqdm.auto import trange

from flashrag.generator.utils import resolve_max_tokens

try:
    from llm.predict.predictor import (
        ModelArgument,
        PredictorArgument,
        batchfy_text,
        create_predictor,
    )
except ImportError:
    print("Please clone and add PaddleNLP to your PYTHONPATH, e.g., `export PYTHONPATH=$PYTHONPATH:/home/your_name/PaddleNLP")

class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.model_path = config["generator_model_path"]

        self.max_input_len = config["generator_max_input_len"]
        self.batch_size = config["generator_batch_size"]
        self.device = config["device"]
        self.gpu_num = paddle.device.cuda.device_count()

        self.config = config
        self.generation_params = config["generation_params"]

    def generate(self, input_list: list) -> List[str]:
        """Get responses from the generator.

        Args:
            input_list: it contains input texts, each item represents a sample.

        Returns:
            list: contains generator's response of each input sample.
        """
        pass


class EncoderDecoderGenerator(BaseGenerator):
    """Class for encoder-decoder model"""

    def __init__(self, config):
        super().__init__(config)
        self.fid = config["use_fid"]
        model_config = AutoConfig.from_pretrained(self.model_path)
        arch = model_config.architectures[0].lower()
        if "t5" in arch:
            if self.fid:
                from flashrag.generator.fid import FiDT5

                self.model = FiDT5.from_pretrained(self.model_path)
            else:
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        else:
            if self.fid:
                assert False, "FiD only support T5"
            self.model = BartForConditionalGeneration.from_pretrained(self.model_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def encode_passages(self, batch_text_passages: List[List[str]]):
        passage_ids, passage_masks = [], []
        for k, text_passages in enumerate(batch_text_passages):
            p = self.tokenizer.batch_encode_plus(
                text_passages,
                max_length=self.max_input_len,
                pad_to_max_length=True,
                return_tensors="pd",
                truncation=True,
            )
            passage_ids.append(p["input_ids"][None])
            passage_masks.append(p["attention_mask"][None])

        passage_ids = paddle.concat(x=passage_ids, axis=0)
        passage_masks = paddle.concat(x=passage_masks, axis=0)
        return passage_ids, passage_masks.bool()

    @paddle.no_grad()
    def generate(self, input_list: List, batch_size=None, **params):
        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = [
                StopWordCriteria(
                    tokenizer=self.tokenizer,
                    prompts=input_list,
                    stop_words=stop_sym,
                )
            ]
            generation_params["stopping_criteria"] = stopping_criteria

        generation_params = resolve_max_tokens(
            params, generation_params, prioritize_new_tokens=True
        )

        responses = []
        for idx in trange(0, len(input_list), batch_size, desc="Generation process: "):
            batched_prompts = input_list[idx : idx + batch_size]
            if self.fid:
                # assume each input in input_list is a list, contains K string
                input_ids, attention_mask = self.encode_passages(batched_prompts)
                inputs = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }
            else:
                inputs = self.tokenizer(
                    batched_prompts,
                    return_tensors="pd",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_len,
                ).to(self.device)

            # TODO: multi-gpu inference
            outputs = self.model.generate(**inputs, **generation_params)

            outputs = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            responses += outputs

        return responses


class PDCausalLMGenerator(BaseGenerator):
    """Class for decoder-only generator."""

    def __init__(self, config, model=None):
        super().__init__(config)
        self.config = config
        lora_path = (
            None
            if "generator_lora_path" not in config
            else config["generator_lora_path"]
        )
        self.model, self.tokenizer = self._load_model(model=model)
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        print(self.generation_config)
        self.use_lora = False
        if lora_path is not None:
            from paddlenlp.peft import LoRAConfig, LoRAModel

            config = LoRAConfig.from_pretrained(lora_path)
            self.model = LoRAModel.from_pretrained(
                self.model, lora_path, lora_config=config
            )
            self.use_lora = True
            self.model.mark_only_lora_as_trainable()

    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator."""
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                # convert_from_torch=True,
                # torch_dtype="auto",
                # device_map="auto",
                # trust_remote_code=True,
            )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if "qwen" not in self.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    def add_new_tokens(
        self, token_embedding_path, token_name_func=lambda idx: f"[ref{idx+1}]"
    ):
        del self.model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        # get original embedding weight matrix
        embedding_layer = self.model.get_input_embeddings()
        embedding_weights = embedding_layer.weight
        original_vocab_size, embedding_dim = embedding_weights.shape

        new_tokens_weights = paddle.load(token_embedding_path)
        new_tokens_length = new_tokens_weights.shape[0]

        # expand vocabulary
        new_tokens = [token_name_func(idx) for idx in range(new_tokens_length)]
        self.tokenizer.add_tokens(new_tokens)

        # create new embedding matrix
        new_vocab_size = original_vocab_size + new_tokens_length
        new_embedding_weights = paddle.zeros(shape=[new_vocab_size, embedding_dim])

        # copy original embeddings to the new weights
        new_embedding_weights[:original_vocab_size, :] = embedding_weights

        # append virtual token embeddings to the new weights
        for token, embedding in zip(new_tokens, new_tokens_weights):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            new_embedding_weights[token_id] = embedding

        # update the embedding table
        # note: we should avoid using the function resize_token_embeddings() because this function will also change the lm_head of the model
        embedding_layer.weight.data = new_embedding_weights
        self.model.eval()

    @paddle.no_grad()
    def generate(
        self,
        input_list: List[str],
        batch_size=None,
        return_scores=False,
        return_dict=False,
        **params,
    ):
        """Generate batches one by one. The generated content needs to exclude input."""

        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = StopWordCriteria(
                tokenizer=self.tokenizer,
                prompts=input_list,
                stop_words=stop_sym,
            )
            generation_params["stopping_criteria"] = stopping_criteria

        generation_params = resolve_max_tokens(
            params, generation_params, prioritize_new_tokens=True
        )

        # set eos token for llama
        if "llama" in self.model_name.lower():
            extra_eos_tokens = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if "eos_token_id" in generation_params:
                generation_params["eos_token_id"].extend(extra_eos_tokens)
            else:
                generation_params["eos_token_id"] = extra_eos_tokens

        responses = []
        scores = []
        generated_token_ids = []
        generated_token_logits = []
        for idx in trange(0, len(input_list), batch_size, desc="Generation process: "):
            paddle.device.cuda.empty_cache()
            batched_prompts = input_list[idx : idx + batch_size]
            inputs = self.tokenizer(
                batched_prompts,
                return_tensors="pd",
                padding=True,
                truncation=True,
                max_length=self.max_input_len,
            )
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                output_scores=True,
                return_dict_in_generate=True,
                **generation_params,
            )
            for i, generated_sequence in enumerate(outputs[0]):
                text = self.tokenizer.decode(
                    generated_sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                new_text = text
                if stop_sym is not None:
                    strip_stopword = True
                    # Find the first occurrence of any stop word
                    lower_stop_index = len(text)  # Default to end of text
                    for sym in stop_sym:
                        stop_index = text.find(sym)
                        if stop_index != -1:
                            # Adjust stop index based on whether we're stripping the stop word
                            stop_index += 0 if strip_stopword else len(sym)
                            lower_stop_index = min(stop_index, lower_stop_index)

                    # Cut the text at the first stop word found (if any)
                    new_text = new_text[:lower_stop_index]
                responses.append(new_text)

        if return_dict:
            generated_token_ids = paddle.cat(generated_token_ids, axis=0)
            generated_token_logits = paddle.cat(generated_token_logits, axis=0)
            return {
                "generated_token_ids": generated_token_ids,
                "generated_token_logits": generated_token_logits,
                "responses": responses,
                "scores": scores,
            }

        if return_scores:
            return responses, scores
        else:
            return responses

    @paddle.no_grad()
    def cal_gen_probs(self, prev, next):
        input_ids = self.tokenizer.encode(prev, add_special_tokens=False)
        target_ids = self.tokenizer.encode(next, add_special_tokens=False)
        context_ids = input_ids + target_ids
        context_tensor = paddle.to_tensor([context_ids]).to(self.device)
        with paddle.no_grad():
            outputs = self.model(context_tensor)
            logits = outputs.logits
            logits = logits[0, len(input_ids) - 1 : len(context_ids) - 1, :]
            logits = logits.to("float32").detach().cpu()
            # softmax to normalize
            probs = paddle.nn.functional.softmax(x=logits, axis=-1)
            # obtain probs of target_ids
            target_probs = probs[range(len(target_ids)), target_ids].numpy()

        return logits, target_probs


class PaddleParallelCausalLMGenerator(BaseGenerator):
    """Class for decoder-only generator."""

    def __init__(self, config, model=None):
        super().__init__(config)
        parser = PdArgumentParser((PredictorArgument, ModelArgument))
        predictor_args, model_args = parser.parse_dict(config.final_config)

        predictor_args.model_name_or_path = config["model2path"][config.generator_model]
        predictor_args.batch_size = config["generator_batch_size"]

        paddle.set_device(predictor_args.device)
        paddle.set_default_dtype(predictor_args.dtype)

        tensor_parallel_degree = paddle.distributed.get_world_size()
        if tensor_parallel_degree > 1:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)

        self.predictor = create_predictor(predictor_args, model_args)
        self.model_path = predictor_args.model_name_or_path
        if "Chat" not in predictor_args.model_name_or_path and "Instruct" not in predictor_args.model_name_or_path:
            self.predictor.tokenizer.chat_template = None

    def add_new_tokens(
        self, token_embedding_path, token_name_func=lambda idx: f"[ref{idx+1}]"
    ):
        del self.predictor.model
        self.predictor.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        # get original embedding weight matrix
        embedding_layer = self.predictor.model.get_input_embeddings()
        embedding_weights = embedding_layer.weight
        original_vocab_size, embedding_dim = embedding_weights.shape

        new_tokens_weights = paddle.load(token_embedding_path)
        new_tokens_length = new_tokens_weights.shape[0]

        # expand vocabulary
        new_tokens = [token_name_func(idx) for idx in range(new_tokens_length)]
        self.predictor.tokenizer.add_tokens(new_tokens)

        # create new embedding matrix
        new_vocab_size = original_vocab_size + new_tokens_length
        new_embedding_weights = paddle.zeros(shape=[new_vocab_size, embedding_dim])

        # copy original embeddings to the new weights
        new_embedding_weights[:original_vocab_size, :] = embedding_weights

        # append virtual token embeddings to the new weights
        for token, embedding in zip(new_tokens, new_tokens_weights):
            token_id = self.predictor.tokenizer.convert_tokens_to_ids(token)
            new_embedding_weights[token_id] = embedding

        # update the embedding table
        # note: we should avoid using the function resize_token_embeddings() because this function will also change the lm_head of the model
        embedding_layer.weight.set_value(new_embedding_weights)
        self.predictor.model.eval()

    @paddle.no_grad()
    def generate(
        self,
        input_list: List[str],
        batch_size=None,
        return_scores=False,
        return_dict=False,
        **params,
    ):
        """Generate batches one by one. The generated content needs to exclude input."""
        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        batch_source_texts = batchfy_text(input_list, batch_size)

        responses = []
        for bs, batch_source_text in enumerate(batch_source_texts):
            outputs = self.predictor.predict(batch_source_text)

            if paddle.distributed.get_rank() == 0:
                for output in outputs:
                    responses.append(output)
        return responses
