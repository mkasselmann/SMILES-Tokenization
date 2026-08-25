import subprocess
import sys
import time
import argparse

# Parameter für Standard-Trie: (k, freq_thr)
TRIE_GRID = [
    {"k": k, "freq_thr": f}
    for k in [6, 8, 10, 12]
    for f in [2, 4, 8]
]

# Parameter für TTG-Trie: (k, freq_thr, entropy_thr)
TTG_GRID = [
    {"k": k, "freq_thr": f, "entropy_thr": e}
    for k in [6, 8, 10, 12]
    for f in [2, 4, 8]
    for e in [1.0, 1.5, 2.0, 2.5]
]

# Parameter für SPE: (num_symbols, min_frequency, augmentation)
SPE_GRID = [
    {"num_symbols": n, "min_frequency": m, "augmentation": a}
    for n in [5000, 10000, 20000]
    for m in [500, 1000]
    for a in [0]
]

# Parameter für APE: (max_vocab_size, min_freq_for_merge)
APE_GRID = [
    {"max_vocab_size": v, "min_freq_for_merge": m}
    for v in [500, 1000, 2000]
    for m in [400, 800]
]

def run_command(cmd, dry_run=False):
    """Führt ein Python-Modul im Terminal aus und misst die Zeit."""
    print(f"\n Ausführen: {' '.join(cmd)}")
    if dry_run:
        return True
    
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        print(f" Fertig in {time.time() - t0:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f" FEHLER bei Ausführung ({e})")
        return False
    except KeyboardInterrupt:
        print("\n\nAbbruch durch Benutzer (Ctrl+C).")
        sys.exit(1)

def run_trie_experiments(dry_run=False):
    print("=" * 60)
    print(f" Starte Standard-TRIE-Experimente ({len(TRIE_GRID)} Konfigurationen)")
    print("=" * 60)
    for params in TRIE_GRID:
        cmd = [
            sys.executable, "-m", "train.train_trie",
            "-k", str(params["k"]),
            "-f", str(params["freq_thr"])
        ]
        run_command(cmd, dry_run=dry_run)

def run_ttg_experiments(dry_run=False):
    print("=" * 60)
    print(f" Starte TTG-Experimente ({len(TTG_GRID)} Konfigurationen)")
    print("=" * 60)
    for params in TTG_GRID:
        cmd = [
            sys.executable, "-m", "train.train_ttg",
            "-k", str(params["k"]),
            "-f", str(params["freq_thr"]),
            "-e", str(params["entropy_thr"])
        ]
        run_command(cmd, dry_run=dry_run)

def run_spe_experiments(dry_run=False):
    print("=" * 60)
    print(f" Starte SPE-Experimente ({len(SPE_GRID)} Konfigurationen)")
    print("=" * 60)
    for params in SPE_GRID:
        cmd = [
            sys.executable, "-m", "train.train_spe",
            "-n", str(params["num_symbols"]),
            "-m", str(params["min_frequency"]),
            "-a", str(params["augmentation"])
        ]
        run_command(cmd, dry_run=dry_run)

def run_ape_experiments(dry_run=False):
    print("=" * 60)
    print(f" Starte APE-Experimente ({len(APE_GRID)} Konfigurationen)")
    print("=" * 60)
    for params in APE_GRID:
        cmd = [
            sys.executable, "-m", "train.train_ape",
            "-v", str(params["max_vocab_size"]),
            "-m", str(params["min_freq_for_merge"])
        ]
        run_command(cmd, dry_run=dry_run)

def main():
    parser = argparse.ArgumentParser(description="Automatische Trainingsläufe mit variierenden Parametern.")
    parser.add_argument(
        "--only", 
        choices=["trie", "ttg", "spe", "ape"], 
        help="Nur einen bestimmten Tokenizer trainieren (Standard: alle vier nacheinander)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Befehle nur anzeigen, aber nicht wirklich ausführen"
    )
    args = parser.parse_args()

    total_start = time.time()

    if args.only == "trie" or args.only is None:
        run_trie_experiments(dry_run=args.dry_run)

    if args.only == "ttg" or args.only is None:
        run_ttg_experiments(dry_run=args.dry_run)
        
    if args.only == "spe" or args.only is None:
        run_spe_experiments(dry_run=args.dry_run)
        
    if args.only == "ape" or args.only is None:
        run_ape_experiments(dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print(f" Alle gewählten Experimente abgeschlossen in {time.time() - total_start:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()