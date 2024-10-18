import paddle
import json
import paddlenlp.transformers as transformers
import paddlenlp.datasets as datasets

def load_model(model_path: str, use_fp16: bool=False):
    model_config = transformers.AutoConfig.from_pretrained(model_path,
        trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained(model_path, convert_from_torch=True)
    model.eval()
    if use_fp16:
        model = model.astype(dtype='float16')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, convert_from_torch=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method='mean'):
    if pooling_method == 'mean':
        if attention_mask is not None:
            attention_mask = attention_mask.astype(last_hidden_state.dtype)  # 确保类型一致
            last_hidden = last_hidden_state.masked_fill(mask=~attention_mask[..., None].astype(dtype='bool'), value=0.0)
            return last_hidden.sum(axis=1) / attention_mask.sum(axis=1)[..., None]
        else:
            raise ValueError("attention_mask must be provided for mean pooling.")
    elif pooling_method == 'cls':
        return last_hidden_state[:, 0]
    elif pooling_method == 'pooler':
        return pooler_output
    else:
        raise NotImplementedError('Pooling method not implemented!')



def load_corpus(corpus_path: str):
    # corpus = datasets.load_dataset('json', data_files=corpus_path, splits=
    #     'train',cache_dir="./cache_pd_dataset")
    corpus = datasets.load_dataset('json', data_files=corpus_path, splits='train')
    return corpus


def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)
            yield new_item


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results
