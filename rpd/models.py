"""Backbone loading and the two model-specific conventions RPD depends on.

LLaDA reads the logits at the masked position itself; Dream is trained with a
one-position shift, so its readout uses the preceding source position. Both are
run in bfloat16 with greedy commits and no KV cache.
"""
import torch
from transformers import AutoModel, AutoTokenizer

CHECKPOINTS = {
    'llada': 'GSAI-ML/LLaDA-8B-Instruct',
    'dream': 'Dream-org/Dream-v0-Instruct-7B',
}


def load(name, checkpoint=None, device='cuda:0', revision=None):
    """Load a masked-diffusion backbone and its tokenizer."""
    if name not in CHECKPOINTS:
        raise ValueError(f'unknown model {name!r}; expected one of {sorted(CHECKPOINTS)}')
    path = checkpoint or CHECKPOINTS[name]
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, revision=revision, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    if model.config.mask_token_id is None:
        raise ValueError('backbone does not expose config.mask_token_id')
    return model, tokenizer


def build_prompt(tokenizer, question):
    """Zero-shot chat-formatted prompt, matching the reported setting."""
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': question}],
        tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=True)['input_ids']
