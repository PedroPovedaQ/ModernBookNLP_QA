
from pdnc_dataset import *
from model import *
from loss import SALoss

from torch.utils.data import Dataset
import json
from safetensors.torch import load_file
import torch
import os, glob
from argparse import ArgumentParser
import numpy as np 
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
import numpy, random
import yaml
from transformers import get_cosine_schedule_with_warmup
from pathlib import Path

# Directory containing this script (e.g. root/preprocess)
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root is one level up from preprocess/
ROOT_DIR = SCRIPT_DIR.parent



def to_device(data: dict, device: torch.device) -> dict:
    """
    Move all tensors in a dictionary to the given device.
    Supports values that are tensors, lists/tuples of tensors,
    or nested dicts. Non-tensor values are left untouched.
    """
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in v
            )
        elif isinstance(v, dict):
            out[k] = to_device(v, device)
        else:
            out[k] = v.to(device)
    return out

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**16
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

from tqdm.auto import tqdm

def run_one_epoch(model, dataloader, optimizer, loss_fn, lr_scheduler) : 
    total_qa_loss = 0
    total_ana_loss = 0
    model.train()
    pbar = tqdm(dataloader)
    
    for idx, batch in enumerate(pbar) :
        with accelerator.accumulate(model):
            labels = batch.pop('candidate_labels')
            labels = [l.float() for l in labels]
            with accelerator.autocast():
                out_ = model(**batch)

            out = out_['qa_scores']
            for o in out:
                assert torch.isfinite(o).all(), "NaN/Inf already in raw scores, before the loss!"
            

            loss = loss_fn(out, labels)
            assert torch.isfinite(loss), "NaN/Inf only appears in the loss, not the scores"
            
            total_qa_loss += loss.detach().cpu().item()
            if 'ana_loss' in out_ :
                loss = loss + out_['ana_loss']
                total_ana_loss+=out_['ana_loss'].detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            lr_scheduler.step()

        pbar.set_description(f'train_loss: {total_qa_loss/(idx+1):.2f} | train_ana_loss: {total_ana_loss/(idx+1):.2f}')
        
    return total_qa_loss / len(dataloader)


def gather_pred_and_labels(g, scores, labels)  : 
    prev = 0 
    out_preds = []
    out_labels = []
    out_ids = []
    for cnt, (s,l) in enumerate(zip(scores, labels)) : 

        try : 
            preds = s.argmax(1)
        except : 
            continue
        preds = g.corefs[g.batch_m==cnt][preds]
        labels = g.speakers[g.batch_q==cnt][g.is_pred[cnt]]
        qids = [g.quote_ids[cnt][i] for i in g.is_pred[cnt]]
        out_ids.extend(qids)
        out_preds.extend(preds.tolist())
        out_labels.extend(labels.tolist())

    return out_preds, out_labels, out_ids


from collections import defaultdict

def majority_vote(preds: Dict) :
    out = {}
    for i,p in preds.items() : 
        if len(p) <= 2 : 
            out[i] = p[0]
        else : 
            out[i] = max(p,key=p.count)

    return out 
@torch.no_grad()
def eval_one(model, dataloader, loss_fn, return_preds=False) : 
    model.eval()
    total_qa_loss = 0 
    total_ana_loss = 0
    for n,p in model.named_modules() : 
        if isinstance(p, torch.nn.Linear) : 
            device = p.weight.device
    all_preds =  defaultdict(list)
    all_labels = defaultdict(list)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader) )
    for idx, batch in pbar: 

        batch = to_device(batch, device) #{k:v.to(device) for k,v in batch.items()}
        init_labels = batch.pop('candidate_labels')
        with accelerator.autocast():
            out_ = model(**batch)
        out = out_['qa_scores']
        out = [o.float() for o in out]
        # if idx == 1 : 
        #     print(out)
        if 'ana_loss' in out_ : 
            total_ana_loss += out_['ana_loss']
        preds, labels, ids = gather_pred_and_labels(batch['g'], out, init_labels)
        for i,p,l in zip(ids, preds, labels) : 
            all_preds[i].append(p)
            all_labels[i].append(l)

        init_labels = [l.float() for l in init_labels]
        loss = loss_fn(out, init_labels)#, batch['repeat_values'])
        total_qa_loss += loss.item()
        pbar.set_description('validation')

    all_preds = majority_vote(all_preds)
    all_labels = majority_vote(all_labels)

    all_p = [all_preds[k] for k in all_preds]
    all_l = [all_labels[k] for k in all_preds]
    accuracy = (np.asarray(all_p) == np.asarray(all_l)).mean()
    if return_preds : 
        return total_qa_loss / len(dataloader), accuracy, (all_preds, all_labels)

    return total_qa_loss / len(dataloader), total_ana_loss / len(dataloader), accuracy

from functools import partial

if __name__=='__main__' : 
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='results/pdnc/')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--grad_acc', default=1, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr', default=7e-5, type=float)
    parser.add_argument('--drop_p', default=0, type=float)
    parser.add_argument('--name_key', default='Components_N2000_S512_K200_ModernBERT_Large.pkl')
    parser.add_argument('--save_path', default='results/')
    parser.add_argument('--anaphora_pred', action='store_true')
    parser.add_argument('--is_direct', action='store_true')
    parser.add_argument('--model_save_path', default='model_ckpts/', type=str)

    args = parser.parse_args()
    config = yaml.safe_load(open(f'{SCRIPT_DIR}/pdnc_config.yaml'))

    folder_name = os.path.split(args.save_path)[-1]
    if len(folder_name) == 0 :
        f1 = os.path.split(args.save_path)[0]
        folder_name= os.path.split(f1)[1]

    device = torch.device('cuda:0')
    save_path = f'results_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_drop{args.drop_p}'

    save_path = os.path.join(SCRIPT_DIR, args.save_path, save_path)

    print(f'Saving at {save_path}')
    os.makedirs(save_path, exist_ok=True)

    accelerator = Accelerator(mixed_precision='bf16')
    
    for split in range(5) : 
        
        if args.is_direct: 
            dataset_cls = IndividualPDNCDataset
        else : 
            dataset_cls = PDNCDataset

        train_dataset = dataset_cls(
            graph_path = f'{ROOT_DIR}/data/pdnc_source',
            split='train',
            split_num=split,
            filter=True,
            name_key=args.name_key
        )
        val_dataset = dataset_cls(
            graph_path = f'{ROOT_DIR}/data/pdnc_source',
            split='val',
            split_num=split,
            filter=False,
            name_key=args.name_key
        )
        test_dataset =  dataset_cls(
            graph_path = f'{ROOT_DIR}/data/pdnc_source',
            split='test',
            split_num=split,
            filter=False,
            name_key=args.name_key
        )

        print(f'Dataset {train_dataset} Size: {len(train_dataset)}')
        if 'longformer' in config['model']['model_id'].lower() : 
            pad_token_id = 1
        else : 
            pad_token_id= 50283
        collate_fn_ = partial(collate_fn, data_args={'pad_token_id' :pad_token_id })

        torch.manual_seed(2026)
        
        g = torch.Generator()
        g.manual_seed(split)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_, num_workers=0, worker_init_fn=seed_worker, generator=g)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_, num_workers=0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_, num_workers=0)

        config['model']['anaphora_pred'] = False
        model = Baseline(config['model'])

        print(model)
        numptrain = sum([p.numel() for p in model.parameters() if p.requires_grad])
        nump = sum([p.numel() for p in model.parameters()])

        print(f'Model Number of Parameters: {nump}')
        print(f'Model Number of Trainable Parameters: {numptrain}')
        

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = SALoss()

        num_warmup_epoch = 1
        num_epoch = 10

        num_warmup_steps = len(train_dataloader) * num_warmup_epoch
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps= len(train_dataloader) * num_epoch
    )
        train_dataloader, model, optimizer, loss_fn, lr_scheduler = accelerator.prepare(train_dataloader, model, optimizer, loss_fn, lr_scheduler)

        unwrapped_loss_fn = SALoss()
        
        for epoch in range(num_epoch) : 
            
            best_acc = 0 
            
            train_loss = run_one_epoch(model, train_dataloader, optimizer, loss_fn, lr_scheduler)
            val_loss, val_ana_loss, val_acc = eval_one(accelerator.unwrap_model(model), val_dataloader, unwrapped_loss_fn)
            
            if val_acc > best_acc : 
                best_ckpt = accelerator.unwrap_model(model).state_dict().copy()
                best_acc = val_acc
                accelerator.save_state(f'{args.model_save_path}/{folder_name}/split_{split}')


            print(f'[Epoch {epoch+1}/{num_epoch}] Train Loss: {train_loss:.2f} Val Loss: {val_loss:.2f} Val Ana Loss {val_ana_loss:.2f} Val Accuracy: {val_acc:.3f}')
        
        model = accelerator.unwrap_model(model)
        model.load_state_dict(best_ckpt)


        test_loss, test_acc, (test_preds, test_labels) = eval_one(model, test_dataloader, unwrapped_loss_fn, return_preds=True)
        
        print(f'[SPLIT {split+1}] Test # Items: {len(test_labels)} Test Loss: {test_loss:.2f} Test Accuracy: {test_acc:.3f}')


        with open(os.path.join(save_path, f'split_{split}.json'), 'w') as f :
            json.dump({
                'accuracy' : test_acc,
                'loss' : test_loss,
                'preds' : test_preds,
                'labels' : test_labels,
            }, f)
        
