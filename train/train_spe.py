import codecs
import pyarrow.dataset as ds, pyarrow as pa, pyarrow.parquet as pq, os
from SmilesPE.learner import *
from experiments.utils import iter_smiles
import time

SLICE = "../data/chebi_smiles.parquet"
OUT = 'spe_chebi.txt'

SMILES = list(iter_smiles(SLICE))
print('Number of SMILES:', len(SMILES))

output = codecs.open(OUT, 'w')
t0 = time.time()
learn_SPE(SMILES, output, 30000, min_frequency=2000, augmentation=1, verbose=True, total_symbols=True)
print(f"✔ SPE saved → {OUT}  ({time.time()-t0:.1f}s)")
