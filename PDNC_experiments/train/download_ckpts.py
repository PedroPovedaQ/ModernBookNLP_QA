from huggingface_hub import hf_hub_download
import os 
MODELPATH = 'ckpts/'
REPO_ID = open('.repo_id').read().strip()

hf_hub_download(repo_id=REPO_ID, filename="ModernBERT_T2000.safetensors", local_dir=MODELPATH)
hf_hub_download(repo_id=REPO_ID, filename="Direct_ModernBERT_T2000.safetensors", local_dir=MODELPATH)