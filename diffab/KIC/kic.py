from diffab.KIC._kic import _kic

import os
import shutil
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
from datetime import datetime
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser(description='use kic to generate different pose')
    parser.add_argument('--json_dir', type=str, required=True, help='json dir')
    parser.add_argument('--split', type=str, required=True, help='test, val, train')
    parser.add_argument('--ncpu', type=int, default=8, help='nums of worker')
    parser.add_argument('--refresh', action='store_true', help='refresh')
    return parser.parse_args()

def split_list_into_sublists(input_list, n):
    avg = len(input_list) // n
    remain = len(input_list) % n
    sublists = []
    i = 0

    for _ in range(n):
        if remain:
            sublists.append(input_list[i : i + avg + 1])
            i += avg + 1
            remain -= 1
        else:
            sublists.append(input_list[i : i + avg])
            i += avg
    return sublists     

def kic(args):
    start_time = datetime.now()
    json_file_path = os.path.join(args.json_dir, args.split+'.json')

    kic_dir = os.path.join(args.json_dir, args.split, 'pdb_kic')
    if args.refresh:
        if os.path.exists(kic_dir):
            print(f"refresh: {args.refresh}, rm {kic_dir}")
            shutil.rmtree(kic_dir)
        else:
            print(f"refresh: {args.refresh}, {kic_dir} not exists! error.")
            return
        os.mkdir(kic_dir)
    if not os.path.exists(kic_dir):
        os.mkdir(kic_dir)

    with open (json_file_path, 'r') as fin:
        lines = fin.read().strip().split('\n')

    pdbfilepaths = []
    chain_names = []
    outfilepaths = []

    cnt = 0
    for line in tqdm(lines):
        item = json.loads(line)
        cnt += 1
        pdb = item['pdb']
        heavy_chain_name = item['heavy']
        light_chain_name = item['light']
        antigen_chain_name = item['antigen']

        pdbfilepath = os.path.join(args.json_dir, args.split, 'pdb', pdb+'_'+heavy_chain_name+'_'+light_chain_name+'_'+antigen_chain_name)
        if not os.path.exists(pdbfilepath) or len(os.listdir(pdbfilepath)) != 100:
            print(pdb+'_'+heavy_chain_name+'_'+light_chain_name+'_'+antigen_chain_name+' file not exists')
            continue
        outfilepath = os.path.join(kic_dir, pdb+'_'+heavy_chain_name+'_'+light_chain_name+'_'+antigen_chain_name)
        if not os.path.exists(outfilepath):
            os.mkdir(outfilepath)
        if not args.refresh:
            l = os.listdir(outfilepath)
            if len(l) < 100:
                shutil.rmtree(outfilepath)
                os.mkdir(outfilepath)
                pdbfilepaths.append(pdbfilepath)
                outfilepaths.append(outfilepath)
                chain_names.append(heavy_chain_name)
            else:
                continue
        else:
            pdbfilepaths.append(pdbfilepath)
            outfilepaths.append(outfilepath)
            chain_names.append(heavy_chain_name)
    if args.refresh:
        print(f'all: {cnt}, valid pdb file: {len(pdbfilepaths)}')
    else:
        print(f"{len(pdbfilepaths)} files need to redo: {[i for i in pdbfilepaths]}")
        if len(pdbfilepaths) == 0:
            return

    cpu_max = multiprocessing.cpu_count()
    ncpu = cpu_max if args.ncpu > cpu_max else args.ncpu
    if cpu_max > args.ncpu :
        print(f"cpu_max: {cpu_max}, use {args.ncpu}")
    else :
        print(f"use cpu_max: {cpu_max}")

    pdbfilepaths_list = split_list_into_sublists(pdbfilepaths, ncpu)
    chain_names_list = split_list_into_sublists(chain_names, ncpu)
    outfilepaths_list = split_list_into_sublists(outfilepaths, ncpu)
    closure_attempt = 500000
    min_solutions = 1
    filter1 = True
    filter2 = False
    mode = 1
    selector_mode = 2
    nums = 100
    
    args_list = [(pdbfilepaths_list[i], outfilepaths_list[i], chain_names_list[i], closure_attempt, \
                  min_solutions, filter1, filter2, mode, selector_mode, nums) for i in range(len(pdbfilepaths_list))]
    if ncpu == 1:
        for a in args_list:
            _kic(a)
    else:
        pool = Pool(processes=ncpu)
        pool.map(_kic, args_list)

    end_time = datetime.now()
    print(f"time: {(end_time-start_time).total_seconds()/(60*60)} h.")

if __name__ == '__main__':
    kic(parse())



