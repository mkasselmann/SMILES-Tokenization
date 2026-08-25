# train_trie.py
import argparse
import time
import os
from experiments.utils import iter_smiles
import train.trie_funcs as tf

SLICE = "data/chebi_smiles.parquet"

def parse_args():
    parser = argparse.ArgumentParser(description="Build and save a Trie compressor.")
    parser.add_argument(
        "-k", "--k", 
        type=int, 
        default=10, 
        help="Max depth / K parameter (default: 10)"
    )
    parser.add_argument(
        "-f", "--freq_thr", 
        type=int, 
        default=4, 
        help="Frequency threshold (default: 4)"
    )
    parser.add_argument(
        "-o", "--out", 
        type=str, 
        default=None, 
        help="Output path for the .pkl file (default: trie_chebi_<K>_<FREQ_THR>.pkl)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    out_path = args.out if args.out else f"trie_chebi_{args.k}_{args.freq_thr}.pkl"

    print(f"Building trie compressor (K={args.k}, freq_thr={args.freq_thr}) …")
    t0 = time.time()
    
    state = tf.prepare_compressor(iter_smiles(SLICE), K=args.k, freq_thr=args.freq_thr)
    tf.save_state(state, out_path)
    
    print(f"✔ Trie saved → {out_path}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
