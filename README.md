# RPD: Reliable Parallel Decoding

Training-free parallel decoding for masked diffusion language models (dLLMs).

RPD makes multi-token commits *reliable*: a position is written only when the
model has visibly settled on it and nothing unresolved to its left could still
change it.

A masked dLLM predicts every masked position at once, but committing several of
them in the same forward pass is only safe when they do not depend on each
other. RPD decides what to commit from two signals read out of the **same**
forward pass, so it adds no extra backbone evaluation:

1. **Suffix progress (SP).** Reading the posterior half of the layer stack, a
   position gets a score `min(K, 6) - w · r`, where `K` is the length of the
   final uninterrupted run of layers whose argmax equals the output token, and
   `r` is the peak-to-final drop of that token's probability inside the run. A
   position that settled early and never wavered scores high.
2. **Cumulative residual entropy.** Scanning the canvas left to right, every
   position that is *not* committed adds its predictive entropy to a shared
   debt. A position may only be committed while the debt accumulated to its left
   stays within a budget `B` (in nats). This is what stops a confident token
   from being written past an unresolved region it may depend on.

A position is committed when its confidence is at least `0.9`, or when its
confidence is at least `0.6` and its SP score clears the model's threshold — and
in both cases only if the entropy gate allows it. If a round admits nothing, the
most confident position within 32 of the frontier is committed so decoding
always terminates.

## Layout

```
rpd/
  decode.py                     decoding loop, candidate admission, defaults
  gate.py                       cumulative residual-entropy gate
  lazy_sp_readout.py            lazy layer projection (default; see below)
  models.py                     backbone loading, prompt formatting
  coupled_open_block_suffix.py  sparse readout over the pending positions
  coupled_gated_suffix.py       SP score: agreement length and peak drawdown
  interlayer_consistency.py     per-layer argmax / probability trajectories
  saturated_consistency.py      saturating agreement-length transform
scripts/
  generate.py                   command-line entry point
```

## Setup

```bash
pip install -r requirements.txt
```

The backbones are downloaded from the Hugging Face Hub on first use:
`GSAI-ML/LLaDA-8B-Instruct` and `Dream-org/Dream-v0-Instruct-7B`. A single GPU
with 48 GB is enough for the reported setting (256 generated positions, no KV
cache).

## Usage

```bash
python scripts/generate.py --model llada \
  --question "Natalia sold clips to 48 friends in April, and half as many in May. How many did she sell?"
```

`--json` prints the per-round record (committed positions, how many came from
each route, whether the fallback fired). To sweep the budget, pass
`--entropy-budget` (4.0 nats is the reported setting; smaller is more
conservative, larger is faster).

In Python:

```python
from rpd import decode
from rpd.decode import config_for
from rpd.models import load, build_prompt

model, tokenizer = load('llada')
prompt_ids = build_prompt(tokenizer, 'Your question here')
result = decode(model, 'llada', prompt_ids, config_for('llada'))
print(tokenizer.decode(result['generated_ids'], skip_special_tokens=True))
print(result['nfe'], 'forward passes')
```

## Reproducing the reported numbers

All results use greedy decoding, 256 generated positions, no KV cache, and the
zero-shot chat template above. The configuration is frozen in
`rpd/decode.py`:

| Parameter | LLaDA | Dream |
|---|---:|---:|
| confidence threshold | 0.9 | 0.9 |
| SP confidence floor | 0.6 | 0.6 |
| SP score threshold | 3.5 | 2.5 |
| drawdown weight `w` | 15.0 | 20.0 |
| agreement cap | 6 | 6 |
| entropy budget `B` | 4.0 nats | 4.0 nats |
| readout layers | posterior half | posterior half |

To evaluate on a benchmark, decode each prompt with `decode()` and score
`result['generated_ids']` with that benchmark's standard metric; `result['nfe']`
is the forward-pass count and `result['seconds']` the wall-clock decoding time.
This repository ships the decoder only — no benchmark data, harness, or result
files are included.

## Notes

- Dream is trained with a one-position shift, so its readout uses the preceding
  source position and its forward is called with `attention_mask='full'`. Both
  are handled in `models.py` and `decode.py`.
- The SP readout adds LM-head projections over the posterior-half layers. These
  cost no extra backbone forward, but they are not free. By default the
  projections are computed lazily: a position whose layer trajectory the entropy
  gate can no longer need is never projected, and the remaining ones are
  screened on the last few layers before the rest are materialised. This is what
  the reported throughput uses. Passing `lazy_readout=False` selects the eager
  reference implementation, which commits exactly the same tokens with exactly
  the same NFE but runs roughly 1.3x slower in wall-clock terms; the two are
  checked against each other in `decode.py`.
- `kappa` and `tau` are retained for readout compatibility and do not affect the
  committed set under the configuration above.
