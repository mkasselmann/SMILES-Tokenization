# train_ape.py
import argparse
import os
import time
from experiments.utils import iter_smiles
from train.ape_tokenizer import APETokenizer

SLICE = "data/chebi_smiles.parquet"  # use make_slice.py once

def parse_args():
    parser = argparse.ArgumentParser(description="Train and save an APE Tokenizer.")
    parser.add_argument(
        "-v", "--max_vocab_size",
        type=int,
        default=1000,
        help="Maximum vocabulary size (default: 1000)"
    )
    parser.add_argument(
        "-m", "--min_freq_for_merge",
        type=int,
        default=800,
        help="Minimum frequency for merge operations (default: 800)"
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        default=None,
        help="Output folder (default: ape_chebi_<max_vocab_size>_<min_freq_for_merge>)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    out_dir = args.out if args.out else f"ape_chebi_{args.max_vocab_size}_{args.min_freq_for_merge}"
    os.makedirs(out_dir, exist_ok=True)

    ape = APETokenizer()
    print(f"Training APE (max_vocab_size={args.max_vocab_size}, min_freq_for_merge={args.min_freq_for_merge}) …")
    t0 = time.time()
    
    ape.train(
        iter_smiles(SLICE),
        max_vocab_size=args.max_vocab_size,
        min_freq_for_merge=args.min_freq_for_merge
    )
    ape.save_pretrained(out_dir)
    print(f"✔ APE saved → {out_dir}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
