import paddle
from typing import List
import numpy as np
from flashrag.retriever.utils import load_model, pooling, parse_query

class Encoder:
    """
    Encoder class for encoding queries using a specified model.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        pooling_method (str): The method used for pooling.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.
        instruction (str): Additional instructions for parsing queries.

    Methods:
        encode(query_list: List[str], is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
    """

    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, instruction):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @paddle.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction)

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pd"
        )
        inputs = {k: v.cuda(blocking=True) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = paddle.zeros(shape=(tuple(inputs["input_ids"].shape)[0], 1), dtype='int64').to(
                inputs['input_ids'].place
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = paddle.ones(inputs['input_ids'].shape, dtype='int64')
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs['attention_mask'], self.pooling_method
            )
        query_emb = paddle.nn.functional.normalize(query_emb, axis=-1)
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order='C')
        return query_emb


class STEncoder:
    """
    STEncoder class for encoding queries using SentenceTransformers.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.
        instruction (str): Additional instructions for parsing queries.

    Methods:
        encode(query_list: List[str], batch_size=64, is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
        multi_gpu_encode(query_list: List[str], is_query=True, batch_size=None) -> np.ndarray:
            Encodes a list of queries into embeddings using multiple GPUs.
    """

    def __init__(self, model_name, model_path, max_length, use_fp16, instruction):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.model = SentenceTransformer(
            model_path, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16 if use_fp16 else torch.float}
        )

    @paddle.no_grad()
    def encode(self, query_list: List[str], batch_size=64, is_query=True) ->np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction)
        query_emb = self.model.encode(
            query_list, batch_size=batch_size,convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
        )
        query_emb = query_emb.astype(np.float32, order='C')
        return query_emb

    @paddle.no_grad()
    def multi_gpu_encode(self, query_list: List[str], is_query=True, batch_size=None) ->np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction)
        pool = self.model.start_multi_process_pool()
        query_emb = self.model.encode_multi_process(
            query_list,
            pool,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        self.model.stop_multi_process_pool(pool)
        query_emb = query_emb.astype(np.float32, order='C')
        return query_emb
