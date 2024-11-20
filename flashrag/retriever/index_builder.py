import argparse
import json
import os
import shutil
import subprocess
import warnings

import faiss
import numpy as np
import paddle
import paddle.distributed as dist
import paddlenlp.datasets as datasets
from tqdm import tqdm


from flashrag.retriever.utils import load_corpus, load_model, pooling, set_default_instruction


class Index_Builder:
    """A tool class used to build an index used in retrieval."""

    def __init__(
        self,
        retrieval_method,
        model_path,
        corpus_path,
        save_dir,
        max_length,
        batch_size,
        use_fp16,
        use_fast_tokenizer=False,
        pooling_method=None,
        instruction=None,
        faiss_type=None,
        embedding_path=None,
        save_embedding=False,
        faiss_gpu=False,
        use_sentence_transformer=False,
        bm25_backend="bm25s",
    ):
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.use_fast_tokenizer = use_fast_tokenizer
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else "Flat"
        self.instruction = instruction
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.use_sentence_transformer = use_sentence_transformer
        self.bm25_backend = bm25_backend
        self.gpu_num = paddle.distributed.get_world_size()
        # set instruction for encode
        if self.instruction is not None:
            self.instruction = self.instruction.strip() + " "
            print("Set instruction for encoding:", self.instruction)
        else:
            self.instruction = set_default_instruction(
                self.retrieval_method, is_query=False
            )
            if self.instruction == "":
                warnings.warn("Instruction is not set!")
            else:
                warnings.warn(f"Instruction is set to default: {self.instruction}")
        # . config pooling method
        if pooling_method is None:
            try:
                # read pooling method from 1_Pooling/config.json
                pooling_config = json.load(
                    open(os.path.join(self.model_path, "1_Pooling/config.json"))
                )
                for k, v in pooling_config.items():
                    if k.startswith("pooling_mode") and v is True:
                        pooling_method = k.split("pooling_mode_")[-1]
                        if pooling_method == "mean_tokens":
                            pooling_method = "mean"
                        elif pooling_method == "cls_token":
                            pooling_method = "cls"
                        else:
                            # raise warning: not implemented pooling method
                            warnings.warn(
                                f"Pooling method {pooling_method} is not implemented.",
                                UserWarning,
                            )
                            pooling_method = "mean"
                        break
            except:
                print(
                    f"Pooling method not found in {self.model_path}, use default pooling method (mean)."
                )
                # use default pooling method
                pooling_method = "mean"
        else:
            if pooling_method not in ["mean", "cls", "pooler"]:
                raise ValueError(f"Invalid pooling method {pooling_method}.")
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        elif not self._check_dir(self.save_dir):
            warnings.warn(
                "Some files already exists in save dir and may be overwritten.",
                UserWarning,
            )
        self.index_save_path = os.path.join(
            self.save_dir,
            f"{self.retrieval_method}_{self.faiss_type}.index",
        )
        self.embedding_save_path = os.path.join(
            self.save_dir,
            f"emb_{self.retrieval_method}.memmap",
        )
        self.corpus = load_corpus(self.corpus_path)
        print("Finish loading...")

    @staticmethod
    def _check_dir(dir_path):
        """Check if the dir path exists and if there is content."""
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        """Constructing different indexes based on selective retrieval method."""
        if self.retrieval_method == "bm25":
            if self.bm25_backend == "pyserini":
                self.build_bm25_index_pyserini()
            elif self.bm25_backend == "bm25s":
                self.build_bm25_index_bm25s()
            else:
                assert False, "Invalid bm25 backend!"
        else:
            self.build_dense_index()

    def build_bm25_index_pyserini(self):
        """Building BM25 index based on Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)
        shutil.copyfile(self.corpus_path, temp_file_path)
        print("Start building bm25 index...")
        pyserini_args = [
            "--collection",
            "JsonCollection",
            "--input",
            temp_dir,
            "--index",
            self.save_dir,
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
        ]
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)
        shutil.rmtree(temp_dir)
        print("Finish!")

    def build_bm25_index_bm25s(self):
        """Building BM25 index based on bm25s library."""
        import bm25s

        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        corpus = datasets.load_dataset(
            "json", data_files=self.corpus_path, split="train"
        )
        corpus_text = corpus["contents"]
        retriever = bm25s.BM25(corpus=corpus, backend="numba")
        retriever.index(corpus_text)
        retriever.save(self.save_dir, corpus=corpus)
        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(
            corpus_size, hidden_size
        )
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=tuple(all_embeddings.shape),
            mode="w+",
            dtype=all_embeddings.dtype,
        )
        length = tuple(all_embeddings.shape)[0]
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(
                range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"
            ):
                j = min(i + save_batch_size, length)
                memmap[i:j] = all_embeddings[i:j]
        else:
            memmap[:] = all_embeddings

    def st_encode_all(self):
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.batch_size = self.batch_size * self.gpu_num

        sentence_list = [item["contents"] for item in self.corpus]
        sentence_list = [f"{self.instruction}{doc}" for doc in sentence_list]
        all_embeddings = self.encoder.encode(sentence_list, batch_size=self.batch_size)

        return all_embeddings

    @paddle.no_grad()
    def encode_all(self):
        if self.gpu_num > 1:
            print("Use multi gpu!")
            dist.init_parallel_env()
            self.encoder = paddle.DataParallel(layers=self.encoder)
            self.sampler = paddle.io.DistributedBatchSampler(
                dataset=self.corpus,
                batch_size=self.batch_size,
                drop_last=False,
            )
        else:
            self.sampler = paddle.io.BatchSampler(
                dataset=self.corpus,
                batch_size=self.batch_size,
                drop_last=False,
            )

        def collate_fn(x):
            if self.retrieval_method == "e5":
                batch_data = ["passage: " + item["contents"] for item in x]
            else:
                batch_data = [item["contents"] for item in x]
            return self.tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                return_tensors="pd",
                max_length=self.max_length,
            )

        self.dataloader = paddle.io.DataLoader(
            dataset=self.corpus,
            batch_sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=4,
        )

        all_embeddings = []

        for inputs in tqdm(self.dataloader, desc="Inference Embeddings:"):
            # print(inputs)
            if "T5" in type(self.encoder).__name__ or (
                self.gpu_num > 1 and "T5" in type(self.encoder._layers).__name__
            ):
                decoder_input_ids = paddle.zeros(
                    shape=(tuple(inputs["input_ids"].shape)[0], 1), dtype="int64"
                ).to(inputs["input_ids"].place)
                output = self.encoder(
                    **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
                )
                embeddings = output.last_hidden_state[:, 0, :]
            else:
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = paddle.ones(
                        inputs["input_ids"].shape, dtype="int64"
                    )
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(
                    output.pooler_output,
                    output.last_hidden_state,
                    inputs["attention_mask"],
                    self.pooling_method,
                )
                if "dpr" not in self.retrieval_method:
                    embeddings = paddle.nn.functional.normalize(x=embeddings, axis=-1)

            if self.gpu_num > 1:
                batch_embedding = []
                dist.all_gather(batch_embedding, embeddings)
                batch_embedding = paddle.concat(batch_embedding, axis=0)
            else:
                batch_embedding = embeddings

            embeddings = batch_embedding.numpy()
            all_embeddings.append(embeddings)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)
        return all_embeddings

    @paddle.no_grad()
    def build_dense_index(self):
        """Obtain the representation of documents based on the embedding model(BERT-based) and
        construct a faiss index.
        """
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")
        if self.use_sentence_transformer:
            from flashrag.retriever.encoder import STEncoder

            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self.model_path,
                max_length=self.max_length,
                use_fp16=self.use_fp16,
            )
            hidden_size = self.encoder.model.get_sentence_embedding_dimension()
        else:
            self.encoder, self.tokenizer = load_model(
                model_path=self.model_path,
                use_fp16=self.use_fp16,
                use_fast_tokenizer=self.use_fast_tokenizer,
            )
            hidden_size = self.encoder.config.hidden_size
        if self.embedding_path is not None:
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(
                self.embedding_path, corpus_size, hidden_size
            )
        else:
            all_embeddings = (
                self.st_encode_all()
                if self.use_sentence_transformer
                else self.encode_all()
            )
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus
        print("Creating index")
        dim = tuple(all_embeddings.shape)[-1]
        faiss_index = faiss.index_factory(
            dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT
        )
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")


def main():
    parser = argparse.ArgumentParser(description="Creating index.")

    # Basic parameters
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--save_dir", default="indexes/", type=str)

    # Parameters for building dense index
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True)
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--faiss_type", default=None, type=str)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument("--sentence_transformer", action="store_true", default=False)
    parser.add_argument("--bm25_backend", default='bm25s', choices=['bm25s','pyserini'])

    args = parser.parse_args()

    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        use_fast_tokenizer=args.use_fast_tokenizer,
        pooling_method=args.pooling_method,
        instruction=args.instruction,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        use_sentence_transformer=args.sentence_transformer,
        bm25_backend=args.bm25_backend,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
