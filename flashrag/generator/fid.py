# Source: FiD official repo: https://github.com/facebookresearch/FiD
# This software is released under Creative Commons public licenses.

import types
import paddle
import paddlenlp.transformers as transformers
import paddle.nn.functional as F
from paddle import nn


class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].view(kwargs["input_ids"].shape[0], -1)
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].view(kwargs["attention_mask"].shape[0], -1)

        return super(FiDT5, self).forward(**kwargs)

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.shape[1]
            input_ids = input_ids.view(input_ids.shape[0], -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.shape[0], -1)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.shape[1]
        return super().generate(
            input_ids=input_ids.view(input_ids.shape[0], -1),
            attention_mask=attention_mask.view(attention_mask.shape[0], -1),
            max_length=max_length,
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.LayerList(sublayers=block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.shape[1]
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = paddle.concat(scores, axis=2)
        bsz, n_heads, n_layers, _ = scores.shape
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.0)
        scores = scores.sum(axis=[1, 2, 4])
        ntokens = context_mask.sum(axis=[2]) * n_layers * n_heads
        scores = scores / ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(paddle.nn.Layer):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages * passage_length, -1),) + outputs[1:]
        return outputs


class CheckpointWrapper(paddle.nn.Layer):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = paddle.to_tensor([], dtype='float32', place=output[0].place, stop_gradient=False)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = paddle.distributed.fleet.utils.recompute(custom_forward,hidden_states, attention_mask, position_bias)
            output = tuple(x if x.shape != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.LayerList(sublayers=block)
    t5stack.block = block

def transpose_aux_func(dims,dim0, dim1):
    perm = list(range(dims))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm

def cross_attention_forward(
    self,
    input,
    mask=None,
    kv=None,
    position_bias=None,
    past_key_value_state=None,
    head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    """
    This only works for computing cross attention over the input
    """
    assert kv != None
    assert head_mask == None
    assert position_bias != None or self.has_relative_attention_bias

    bsz, qlen, dim = input.shape
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.shape[1]

    q = self.q(input).view(bsz, -1, n_heads, d_heads)
    q = paddle.transpose(q,transpose_aux_func(q.ndim,1,2))
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads)
        k = paddle.transpose(k,transpose_aux_func(k.ndim,1,2))
        v = self.v(kv).view(bsz, -1, n_heads, d_heads)
        v = paddle.transpose(v,transpose_aux_func(v.ndim,1,2))
    else:
        k, v = past_key_value_state

    scores = paddle.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
        scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(x=scores.astype(dtype='float32'),axis=-1).astype(dtype=scores.dtype)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = paddle.matmul(attn, v)
    output = output.transpose(perm=transpose_aux_func(output.ndim, 1, 2))\
                            .contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output
