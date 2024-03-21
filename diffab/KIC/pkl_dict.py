import os
import pickle

from tqdm import tqdm
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()

def run(args):
    splits = []
    if args.train:
        splits.append('train')
    if args.val:
        splits.append('val')
    if args.test:
        splits.append('test')

    d = {}

    for split in splits:
        print(f'start {split}')
        cnt = 0
        valid = 0
        pdb_kic_path = os.path.join(args.path, split, 'pdb_kic')
        dirlist = os.listdir(pdb_kic_path)
        for dir in tqdm(dirlist):
            targetdir = os.path.join(pdb_kic_path, dir)
            cnt += 1
            if os.path.exists(targetdir) and len(os.listdir(targetdir)) == 200:
                d[os.path.basename(targetdir)] = targetdir
                valid += 1

        print(f'cnt: {cnt}, valid: {valid}')

    with open (args.outpath, 'wb') as fout:
        pickle.dump(d, fout)

if __name__ == '__main__':
    run(parse())
    
