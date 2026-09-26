"""Decode a single prompt with RPD.

    python scripts/generate.py --model llada --question "..."
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpd import decode
from rpd.decode import config_for
from rpd.models import load, build_prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['llada', 'dream'], required=True)
    ap.add_argument('--question', required=True)
    ap.add_argument('--checkpoint', default=None, help='override the default HF checkpoint')
    ap.add_argument('--gen-len', type=int, default=256)
    ap.add_argument('--entropy-budget', type=float, default=None,
                    help='residual entropy budget in nats (default: 4.0)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--json', action='store_true', help='print the full record as JSON')
    args = ap.parse_args()

    model, tokenizer = load(args.model, args.checkpoint, args.device)
    overrides = {'gen_len': args.gen_len}
    if args.entropy_budget is not None:
        overrides['entropy_budget'] = args.entropy_budget
    config = config_for(args.model, **overrides)

    prompt_ids = build_prompt(tokenizer, args.question)
    result = decode(model, args.model, prompt_ids, config)
    result['text'] = tokenizer.decode(result['generated_ids'], skip_special_tokens=True)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result['text'].strip())
        print(f"\n[NFE {result['nfe']}  {result['seconds']:.2f}s  "
              f"{len(result['generated_ids']) / result['seconds']:.1f} tok/s]")


if __name__ == '__main__':
    main()
