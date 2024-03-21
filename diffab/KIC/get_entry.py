# we need to get entery first
# such as : XXXX_X_X, we need to get pdb name, file path, H chain name, L chain name, antigen chain name.
from tqdm import tqdm
from diffab.datasets.sabdab import SAbDabDataset
import os
import json

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--chothia_dir', type=str, default='/home/AI4Science/wuj2308/diffab/rectflow_seq_sabdab/data/all_structures/chothia')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print(f'{args.split} set.')

    dataset = SAbDabDataset(
        processed_dir=args.processed_dir,
        split=args.split
    )
    
    info_list = []
    for i in tqdm(range(len(dataset))):
        id = dataset[i]['id']
        pdb, heavy, light, antigen = id.split('_')
        if not heavy:
            continue
        try:
            H3_seq = dataset[i]['heavy']['H3_seq']
        except:
            continue
        info = {
            "pdb": pdb,
            "heavy": heavy,
            "light": light,
            "antigen": antigen,
            "H3_seq": H3_seq,
            "pdb_path": os.path.join(args.chothia_dir, '{}.pdb'.format(pdb))
        }
        info_list.append(info)
    if os.path.exists(os.path.join(args.save_dir, args.split+'.json')):
        print(os.path.join(args.save_dir, args.split+'.json')+' exists, rm.')
        os.remove(os.path.join(args.save_dir, args.split+'.json'))

    fout = open(os.path.join(args.save_dir, args.split+'.json'), 'w')
    for info in info_list:
        info_str = json.dumps(info)
        fout.write(f'{info_str}\n')
    fout.close()
    print('info save in ' + os.path.join(args.save_dir, args.split+'.json'))

