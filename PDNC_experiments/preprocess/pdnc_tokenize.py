# from importlib import reload
import pdnc_to_graph as convert 
import os 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# reload(convert)

import glob, os
from tqdm.auto import tqdm
from multiprocessing import Pool
import pickle
from functools import partial

def tokenize_one(f, aligner, N, S, K, name='ModernBERT_Large') : 
    path = os.path.split(f)[0]


    # if not os.path.exists(f'{path}/New_graph_N{N}_S{S}_K{K}_ModernBERT_Large.pkl') : 
    try : 
        id = (path.split('/')[-1])
        G = convert.build_book_graph(path, id, N=N, S=S, K=K, aligner=aligner)
        with open(f'{path}/Graph_N{N}_S{S}_K{K}_{name}.pkl', 'wb') as f:
            pickle.dump(G, f)
        # with open(f'{path}/New_graph_N{N}_S{S}_K{K}_Longformer_Large.pkl', 'wb') as f:
        #     pickle.dump(G, f)

    except Exception as e:
        print(e)
        return None

        
if __name__ == '__main__' : 
    
    N = 2000
    S = 512
    K = 200
    
    files = glob.glob('data/pdnc_source/*/*.tokens')
    aligner = convert.WordAligner('answerdotai/ModernBERT-large')
    name = 'ModernBERT_Large'

    # aligner = convert.WordAligner('allenai/longformer-base-4096')

    func = partial(tokenize_one, aligner=aligner, N=N, S=S, K=K, name=name)
    with Pool(32) as p : 
        for res in tqdm(p.imap_unordered(func, files), total=len(files)) : 
            pass
