from Bio import PDB
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
import logging
import torch
import multiprocessing
import argparse
import pickle
import os
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from diffab.utils.protein import parsers, constants


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _label_heavy_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map
    
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('H', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    # Add CDR sequence annotations
    data['H1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H1] )
    data['H2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H2] )
    data['H3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H3] )

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

    # Filter: ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map
    
def _label_light_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('L', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    data['L1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L1] )
    data['L2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L2] )
    data['L3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L3] )

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

    # Ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map

def preprocess_structure(task):
    pdbid = task['id']
    pdbpath = task['pdbpath']

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdbpath)[0]

    parsed = {
        'id': pdbid,
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }
    try:
        if task['heavy'] is not None:
            (
                parsed['heavy'],
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                model[task['heavy']],
                max_resseq=113
            ))
        
        if task['light'] is not None:
            (
                parsed['light'],
                parsed['light_seqmap']
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(
                model[task['light']],
                max_resseq = 106    # Chothia, end of Light chain Fv
            ))

        if parsed['heavy'] is None and parsed['light'] is None:
            raise ValueError('Neither valid H-chain or L-chain is found.')
        
        if len(task['antigen']) > 0:
            chains = [model[c] for c in task['antigen']]
            (
                parsed['antigen'], 
                parsed['antigen_seqmap']
            ) = parsers.parse_biopython_structure(chains)
    except (
        PDBExceptions.PDBConstructionException, 
        parsers.ParsingException, 
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'], 
            e.__class__.__name__, 
            str(e)
        ))
        return None
    
    outpath = os.path.join(os.path.dirname(pdbpath), os.path.basename(pdbpath).split('.')[0]+'.pkl')# pdbpath: xxx/x.pdb outpath: xxx/x.pkl
    if os.path.exists(outpath):
        os.remove(outpath)
    
    with open (outpath, 'wb') as fout:
        pickle.dump(parsed, fout)

    return (parsed, pdbpath)

def main():
    task = {
        'id': '3zkq',
        'pdbpath': '/datapool/data2/home/majianzhu/rectflow_seq_sabdab/KICDATA/predictseq/val/pdb_kic/3zkq_D__A/0.pdb',
        'heavy': 'D',
        'light': None,
        'antigen': 'A'
    }
    res = preprocess_structure(task)
    print(res)

main()