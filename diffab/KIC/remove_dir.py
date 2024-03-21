import shutil
import os
from tqdm import tqdm

path = '/datapool/data2/home/majianzhu/rectflow_seq_sabdab/KICDATA/predictseq/train/pdb_kic'

pdblist = os.listdir(path)


for pdb in tqdm(pdblist):
    pdbdirpath = os.path.join(path, pdb)
    if len(os.listdir(pdbdirpath)) == 0:
        shutil.rmtree(pdbdirpath)