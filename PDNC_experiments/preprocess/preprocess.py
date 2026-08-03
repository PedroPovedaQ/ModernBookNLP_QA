import json, os, glob
from argparse import ArgumentParser
from tqdm.auto import tqdm
from multiprocessing import Pool
from pdnc_tokenize import tokenize_one
from pdnc_nx2geo import convert_one
from pdnc_to_graph import WordAligner
from functools import partial
from pathlib import Path

# Directory containing this script (e.g. root/preprocess)
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root is one level up from preprocess/
ROOT_DIR = SCRIPT_DIR.parent

def do_one(ff, N, S, K, restrict=False, name='ModernBERT_Large') :
    path = os.path.split(ff)[0]
    if not os.path.exists(f'{path}/Graph_N{N}_S{S}_K{K}_{name}.pkl') : 
        tokenize_one(ff, aligner, N=N, S=S, K=K, name=name)
    convert_one(f'{path}/Graph_N{N}_S{S}_K{K}_{name}.pkl', N=N, S=S, K=K, restrict=restrict, name=name)


if __name__ =='__main__' : 
    parser = ArgumentParser()
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--S', type=int, default=512)
    parser.add_argument('--K', type=int, default=200)
    parser.add_argument('--restrict', action='store_true')
    parser.add_argument('--model_id', default='answerdotai/ModernBERT-large', type=str)

    args = parser.parse_args()

    files = glob.glob(f'{ROOT_DIR}/data/pdnc_source/*/*.entities')
    aligner = WordAligner(args.model_id)

    if 'ModernBERT' in args.model_id :
        name = 'ModernBERT_Large'   
    elif 'longformer' in args.model_id.lower() :
        name = 'Longformer'
    else : 
        print('[warning] unknown model name. make sure to use the same model when training')
        name = args.model_id.split('/')[-1] 

    func = partial(do_one, N=args.T, S=args.S, K=args.K, name=name, restrict=args.restrict)

    with Pool(16) as p :
        for res in tqdm(p.imap_unordered(func, files), total=len(files)) : 
            pass
