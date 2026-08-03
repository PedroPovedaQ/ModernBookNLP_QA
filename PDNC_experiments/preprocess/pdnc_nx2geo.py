import glob, pickle
import torch_geometric

import networkx as nx 
import torch 
import networkx as nx
import os
from tqdm.auto import tqdm
import pickle
from multiprocessing import Pool
import re 
from functools import partial

def convert_one(ff, N, S, K, restrict=False, name='ModernBERT_Large') : 

    path = os.path.split(ff)[0]

    list_of_graphs = []
    
    with open(ff,'rb') as f:
        G = pickle.load(f)
        connecteds = list(nx.connected_components(G) )

        for idx, component in (enumerate(connecteds)):#, total=len(connecteds)) : 
            subG = G.subgraph(component)
            mapp = {v:k for k,v in enumerate(component)}
            inv_mapp = {k:v for k,v in enumerate(component)}
            subG = nx.relabel_nodes(subG, mapp)
            node_type = []
            input_ids = []
            spans = []
            coref = []
            speakers = []
            quote_ids = []
            to_keep = []
            for v in range(len(subG)) : 
               
                if 'node_type' not in subG.nodes[v] : 
                    subG.nodes[v]['node_type'] = 'quote'
                if subG.nodes[v]['node_type'] == 'mention' : 

                    if restrict : 
                        if subG.nodes[v]['mention_type'] != 'PROPN' : 
                            continue
                            
                    node_type.append(2)
                    spans.append((subG.nodes[v]['st'], subG.nodes[v]['et'] ))
                    coref.append(int(subG.nodes[v]['char_id']))
                    
                elif subG.nodes[v]['node_type'] == 'quote' : 
                    node_type.append(1)
                    
                    if 'Q_' in inv_mapp[v] : 
                        quote_ids.append(re.findall('Q[\d]+\-[\d]', inv_mapp[v]))
                    else : 
                        quote_ids.append(None)
                    if 'st' in subG.nodes[v] : 
                        spans.append((subG.nodes[v]['st'], subG.nodes[v]['et'] ))
                    else : 
                        spans.append((-1,-1))

                    if 'char_id' in subG.nodes[v] :
                        spk_id = subG.nodes[v]['char_id']
                        if spk_id is not None : 
                            speakers.append(int(spk_id))
                        else :
                            speakers.append(-1)
                    else :
                        speakers.append(-1)
                        
                else : 
                    node_type.append(0)
                    spans.append((0,0))
                    
                if 'input_ids' in subG.nodes[v]:
                    input_ids.append(subG.nodes[v]['input_ids'])

                to_keep.append(v)

            subG = subG.subgraph(to_keep)
            
            data = torch_geometric.data.Data(edge_index=torch.LongTensor(list(subG.edges)).t(),num_nodes=len(subG) )
            data.node_types = torch.LongTensor(node_type)
            data.input_ids = torch.LongTensor(input_ids)
            data.spans = torch.LongTensor(spans).t()
            data.corefs = torch.LongTensor(coref)
            data.speakers = torch.LongTensor(speakers)
            data.quote_ids = quote_ids

            list_of_graphs.append(data)
            
        if restrict : 
            out_p=f'Restricted_components_N{N}_S{S}_K{K}_{name}.pkl'
        else : 
            out_p=f'Components_N{N}_S{S}_K{K}_{name}.pkl'

        with open(f'{path}/{out_p}', 'wb') as tgt_f: 
            pickle.dump(list_of_graphs, tgt_f)


if __name__=='__main__' : 
    N = 2000
    S = 512
    K = 200
    RESTRICT = False
        
    files = glob.glob(f'data/pdnc_source/*/Graph_N{N}_S{S}_K{K}_ModernBERT*.pkl')
    name = 'ModernBERT_Large'
    func = partial(convert_one, N=N, S=S, K=K, name=name, restrict=RESTRICT)

    with Pool(32) as p :
        for res in tqdm(p.imap_unordered(func, files), total=len(files)) : 
            pass
