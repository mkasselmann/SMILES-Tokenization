# train_trie.py
import argparse
import os
import time

from experiments.utils import iter_smiles
import train.trie_funcs as tf

SLICE = "data/chebi_smiles.parquet"
OUT_DIR = "ttg_vocab" 


def _make_out_name(k: int, freq: int, ent: float, out_dir: str = OUT_DIR) -> str:
    """
    Produce a file name that encodes the hyper-parameters, e.g.
    ttg_vocab/ttg_chebi_K8_F4_H2p0.pkl
    (the dot in entropy is replaced by “p” to keep the name shell-safe).
    """
    os.makedirs(out_dir, exist_ok=True)
    ent_str = str(ent).replace(".", "p")
    return os.path.join(out_dir, f"ttg_chebi_K{k}_F{freq}_H{ent_str}.pkl")


def run_ttg(k: int = 8, freq_thr: int = 4, entropy_thr: float = 2.0, out_dir: str = OUT_DIR) -> str:
    """
    Build a TTG-guided compressor with the given hyper-parameters.
    Returns the path of the saved pickle.

    Example
    -------
    >>> run_ttg(k=10, freq_thr=3, entropy_thr=1.5)
    'ttg_vocab/ttg_chebi_K10_F3_H1p5.pkl'
    """
    out_path = _make_out_name(k, freq_thr, entropy_thr, out_dir=out_dir)

    print(
        f"Building TTG-guided trie compressor (K={k}, FREQ_THR={freq_thr}, "
        f"ENTROPY_THR={entropy_thr}) …"
    )
    t0 = time.time()

    state = tf.prepare_compressor_with_ttg(
        iter_smiles(SLICE),
        K=k,
        freq_thr=freq_thr,
        entropy_thr=entropy_thr,
    )

    tf.save_state(state, out_path)
    print(f"✔ Trie saved → {out_path}  ({time.time() - t0:.1f}s)")

    # trie_fert, trie_avg, trie_var, trie_ent = compute_trie_metrics(SLICE, state)
    # print(f"✔ Trie Metrics Computed → {out_path}  ({time.time() - t0:.1f}s)")

    return out_path  # , trie_fert, trie_avg, trie_var, trie_ent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save a TTG-guided Trie compressor."
    )
    parser.add_argument(
        "-k", "--k",
        type=int,
        default=8,
        help="Max depth / K parameter (default: 8)",
    )
    parser.add_argument(
        "-f", "--freq_thr",
        type=int,
        default=4,
        help="Frequency threshold (default: 4)",
    )
    parser.add_argument(
        "-e", "--entropy_thr",
        type=float,
        default=2.0,
        help="Entropy threshold (default: 2.0)",
    )
    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_ttg(
        k=args.k,
        freq_thr=args.freq_thr,
        entropy_thr=args.entropy_thr,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()