import pyrosetta
import json
import argparse
import os
from tqdm import tqdm
import multiprocessing
import shutil

def parse():
    parser = argparse.ArgumentParser(description='mutate seq for kic')
    parser.add_argument('--json_dir', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--ncpu', type=int, default=8)
    parser.add_argument('--chothia_dir', type=str, default='/datapool/data2/home/majianzhu/rectflow_seq_sabdab/data/all_structures/chothia')
    parser.add_argument('--refresh', action='store_true')
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

def post(args):
    pyrosetta.init(' '.join([
        '-mute', 'all',
        '-use_input_sc',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy', 'false',
        '-load_PDB_components', 'false',
        '-relax:default_repeats', '2',
        '-no_fconfig',
    ]))
    seq_list, pdbpath_list, pos_list, outpath_list = args
    for seq, pdbpath, pos, outpath in zip(seq_list, pdbpath_list, pos_list, outpath_list):
        if len(seq[0]) != pos[1]-pos[0]+1:
            print(f"{pdbpath} len wrong!")
            continue
        try:
            pose = pyrosetta.pose_from_pdb(pdbpath)
        except:
            print(f'{pdbpath} can not read.')
            continue
        for num in range(len(seq)):
            try:
                for i, j in zip(range(pos[0], pos[1]+1), range(len(seq[num]))):
                    try:
                        pyrosetta.toolbox.mutate_residue(pose, i, seq[num][j])
                    except:
                        print(f'{pdbpath} can not mutate.')
                        raise RuntimeError
                try:
                    pose.dump_pdb(os.path.join(outpath, str(num)+'.pdb'))
                except:
                    print(f'{outpath} dump failed')
                    raise RuntimeError

            except RuntimeError:
                break

def run(args):
    with open (os.path.join(args.json_dir, args.split+'.json')) as fin:
        lines = fin.read().strip().split('\n')

    if not os.path.exists(os.path.join(args.json_dir, args.split)):
        os.mkdir(os.path.join(args.json_dir, args.split))

    if os.path.exists(os.path.join(args.json_dir, args.split,'pdb')) and args.refresh:
        shutil.rmtree(os.path.join(args.json_dir, args.split, 'pdb'))
        os.mkdir(os.path.join(args.json_dir, args.split, 'pdb'))
    elif not os.path.exists(os.path.join(args.json_dir, args.split, 'pdb')) and args.refresh:
        print('should not use refresh here.')
        return
    elif not os.path.exists(os.path.join(args.json_dir, args.split, 'pdb')) and not args.refresh:
        os.mkdir(os.path.join(args.json_dir, args.split, 'pdb'))
    
    seqs = []
    pdbpaths = []
    poss = []
    outpaths = []

    pyrosetta.init(' '.join([
        '-mute', 'all',
        '-use_input_sc',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy', 'false',
        '-load_PDB_components', 'false',
        '-relax:default_repeats', '2',
        '-no_fconfig',
    ]))

    print('starting get info')

    for line in tqdm(lines):
        item = json.loads(line)
        pdb = item['pdb']
        heavy_chain_name = item['heavy']
        light_chain_name = item['light']
        antigen_chain_name = item['antigen']
        h3_init_seq = item['H3_seq']
        seq = item['res_seq']
        save_dir = os.path.join(args.json_dir, args.split, 'pdb', pdb+'_'+heavy_chain_name+'_'+light_chain_name+'_'+antigen_chain_name)

        if not args.refresh and os.path.exists(save_dir) and len(os.listdir(save_dir)) == 100:
            continue
        elif os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
        else:
            os.mkdir(save_dir)

        pdb_file_path = os.path.join(args.chothia_dir, pdb+'.pdb')
        try:
            pose = pyrosetta.pose_from_pdb(pdb_file_path)
        except:
            print(f'can not read {pdb_file_path}')
            shutil.rmtree(save_dir)
            continue
        # cdrh3
        # chothia cdrh3: 95-102
        pos_begin = pose.pdb_info().pdb2pose(heavy_chain_name, 95)
        pos_end = pose.pdb_info().pdb2pose(heavy_chain_name, 102)

        if pos_begin == 0 or pos_end == 0:
            print(f'{pdb_file_path} missing too many residue! can not locate CDRs.')
            shutil.rmtree(save_dir)
            continue

        if len(h3_init_seq) != pos_end - pos_begin +1:
            print(f'{pdb_file_path} len wrong!')
            shutil.rmtree(save_dir)
            continue

        seqs.append(seq)
        pdbpaths.append(pdb_file_path)
        poss.append((pos_begin, pos_end))
        outpaths.append(save_dir)

    if not args.refresh:
        print(f'{len(outpaths)} files need to redo: {pdbpaths}')
        if len(outpaths) == 0:
            return
    
    seqlist = split_list_into_sublists(seqs, args.ncpu)
    pdbpathlist = split_list_into_sublists(pdbpaths, args.ncpu)
    poslist = split_list_into_sublists(poss, args.ncpu)
    outpathlist = split_list_into_sublists(outpaths, args.ncpu)

    args_list = [(seqlist[i], pdbpathlist[i], poslist[i], outpathlist[i]) for i in range(len(seqlist))]
    print(f'cpu max: {multiprocessing.cpu_count()}, use {args.ncpu}')
    pool = multiprocessing.Pool(processes=args.ncpu)
    pool.map(post, args_list)

if __name__ == '__main__':
    run(parse())
