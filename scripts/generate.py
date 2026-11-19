"""Decode a single prompt with RPD.

    python scripts/generate.py --model llada --question "..."

Release status: the decoding loop is withheld pending re-verification, so this
entry point currently exits with NotImplementedError.
"""
import argparse
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
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--gen-len', type=int, default=256)
    ap.add_argument('--entropy-budget', type=float, default=None)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    model, tokenizer = load(args.model, args.checkpoint, args.device)
    overrides = {'gen_len': args.gen_len}
    if args.entropy_budget is not None:
        overrides['entropy_budget'] = args.entropy_budget
    result = decode(model, args.model, build_prompt(tokenizer, args.question),
                    config_for(args.model, **overrides))
    print(tokenizer.decode(result['generated_ids'], skip_special_tokens=True).strip())


if __name__ == '__main__':
    main()
