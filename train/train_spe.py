# train_spe.py
import argparse
import codecs
import time
import os
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from SmilesPE.learner import learn_SPE
from experiments.utils import iter_smiles

SLICE = "data/chebi_smiles.parquet"

def parse_args():
    parser = argparse.ArgumentParser(description="Train and save a SmilesPE (SPE) model.")
    parser.add_argument(
        "-n", "--num_symbols", 
        type=int, 
        default=20000, 
        help="Number of symbols / vocabulary size (default: 20000)"
    )
    parser.add_argument(
        "-m", "--min_frequency", 
        type=int, 
        default=1000, 
        help="Minimum frequency threshold (default: 1000)"
    )
    parser.add_argument(
        "-a", "--augmentation", 
        type=int, 
        default=0, 
        help="Augmentation factor (default: 0)"
    )
    parser.add_argument(
        "-o", "--out", 
        type=str, 
        default=None, 
        help="Output path for the SPE file (default: spe_chebi_<num_symbols>_<min_frequency>_<augmentation>.txt)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    out_path = args.out if args.out else f"spe_chebi_{args.num_symbols}_{args.min_frequency}_{args.augmentation}.txt"

    RDLogger.DisableLog('rdApp.*')
    print("Loading SMILES …")
    all_smiles = list(iter_smiles(SLICE))
    SMILES = [smiles for smiles in all_smiles if Chem.MolFromSmiles(smiles) is not None]
    print(f"Number of SMILES: {len(SMILES)}")
    print(f"Skipped invalid SMILES: {len(all_smiles) - len(SMILES)}")

    print(f"Learning SPE (num_symbols={args.num_symbols}, min_frequency={args.min_frequency}, augmentation={args.augmentation}) …")
    t0 = time.time()
    
    with codecs.open(out_path, 'w', encoding='utf-8') as output:
        learn_SPE(
            SMILES, 
            output, 
            args.num_symbols, 
            min_frequency=args.min_frequency, 
            augmentation=args.augmentation, 
            verbose=True, 
            total_symbols=True
        )
        
    print(f"✔ SPE saved → {out_path}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
