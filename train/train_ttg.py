# train/train_ttg.py
import argparse
import os
import time

from experiments.utils import iter_smiles
import train.trie_funcs as tf

SLICE = "data/pubchem_100K.parquet"
OUT_DIR = "ttg_vocab"


def _make_out_name(k: int, freq: int, ent: float, out_dir: str = OUT_DIR) -> str:
    """
    Erstellt den Ausgabepfad und stellt sicher, dass das Verzeichnis existiert.
    Beispiel: ttg_vocab/ttg_pubchem_K8_F4_H2p0.pkl
    """
    os.makedirs(out_dir, exist_ok=True)
    ent_str = str(ent).replace(".", "p")
    return os.path.join(out_dir, f"ttg_pubchem_K{k}_F{freq}_H{ent_str}.pkl")


def run_ttg(k: int = 8, freq_thr: int = 4, entropy_thr: float = 2.0, out_dir: str = OUT_DIR) -> str:
    """
    Baut den TTG-gestützten Trie-Kompressor und speichert die Datei ab.
    Gibt den Pfad der gespeicherten .pkl Datei zurück.
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
    print(f"✔ TTG Trie saved → {out_path}  ({time.time() - t0:.1f}s)")

    return out_path


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
        help=f"Output directory to save generated files (default: {OUT_DIR})",
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