import random
import os
import argparse
from tqdm import tqdm
import json

def random_generate(len):
    l = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P']
    res_seq = []
    for _ in range(100):
        res_seq.append([random.choice(l) for _ in range(len)])
    return res_seq

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    return parser.parse_args()

def run(args):
    json_path = args.json_path

    with open(json_path, 'r') as f:
        lines = f.read().strip().split('\n')

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    info_list = []

    for line in tqdm(lines):
        item = json.loads(line)
        H3_seq = item["H3_seq"].strip()
        res_seq = random_generate(len(H3_seq))
        info = {
            "pdb": item['pdb'],
            "heavy": item['heavy'],
            "light": item['light'],
            "antigen": item['antigen'],
            "H3_seq": item['H3_seq'],
            "res_seq": res_seq,
            "pdb_path": item['pdb_path']
        }
        info_list.append(info)
    save_json_path = os.path.join(args.save_path, os.path.basename(args.json_path))
    if os.path.exists(save_json_path):
        os.remove(save_json_path)
    fout = open(save_json_path, 'w')
    for info in info_list:
        info_str = json.dumps(info)
        fout.write(f'{info_str}\n')
    fout.close()

    print(f'save res in {save_json_path}.')

if __name__ == '__main__':
    run(parse())