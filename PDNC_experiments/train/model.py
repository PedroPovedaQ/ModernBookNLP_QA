import torch.nn as nn 
from transformers import ModernBertModel
import torch.nn.functional as F 
import torch 
from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available() : 
    ATTN_IMP = 'flash_attention_2'
else :
    ATTN_IMP = 'sdpa'


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    token_embeddings: [B, S, H] - transformer output
    attention_mask:   [B, S]    - 1 for real tokens, 0 for padding
    Returns: [B, H] - mean-pooled sentence embeddings
    """
    # 1. Expand mask to match embedding dimensions: [B, S] -> [B, S, H]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # 2. Zero out embeddings at padding positions, then sum over sequence dim
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)  # [B, H]

    # 3. Count real (non-padded) tokens per sequence, avoid division by zero
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # [B, H]

    # 4. Divide sum by count -> mean over valid tokens only
    mean_pooled = sum_embeddings / sum_mask  # [B, H]

    return mean_pooled


class Baseline(nn.Module) : 
    def __init__(self, config) : 
        super().__init__()
        self.config = config

        if 'longformer' in config['model_id'].lower() : 
            from transformers import LongformerModel
            self.bert = LongformerModel.from_pretrained(
                config['model_id'],
            )
        else : 
            self.bert = ModernBertModel.from_pretrained(
                config['model_id'],
                attn_implementation=ATTN_IMP
    )
        num_bert_layers = self.bert.config.num_hidden_layers

        if not config['bert_start_train_layers'] == 'all' : 

            if  config['bert_start_train_layers'] == 'none'  : 
                for n,p in self.bert.named_parameters() : 
                    p.requires_grad = False
            else : 
                trained_layers = [
                    i for i in range(num_bert_layers + config['bert_start_train_layers'], num_bert_layers)
                ]
                for n,p in self.bert.named_parameters() : 
                    if not any([f'layers.{i}' in n for i in trained_layers]) : 
                        p.requires_grad = False

        self.BH = self.bert.config.hidden_size
        self.virtualH = nn.Embedding(1,self.BH*2).weight

        # here for compatibility, we dont use it
        self.do_anaphora = False
        if self.do_anaphora : 
            self.do_anaphora = True
            self.anaphora_proj =nn.Sequential(
                nn.Linear( self.bert.config.hidden_size*4, self.bert.config.hidden_size*2),
                nn.BatchNorm1d(self.bert.config.hidden_size*2),
                nn.GELU(),
                nn.Linear(self.bert.config.hidden_size*2,1),
        )

        self.proj = nn.Sequential(
            nn.Linear( self.bert.config.hidden_size*4, self.bert.config.hidden_size*2),
            nn.BatchNorm1d(self.bert.config.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.bert.config.hidden_size*2,1),
        )

        self.special_ids = torch.LongTensor([50280, 50282, 50283, 50281, 50284])

    def _get_input_embeddings(self, g, input_ids, att_mask,  st, et) : 
        
        H = self.bert(input_ids=input_ids, attention_mask=att_mask).last_hidden_state

        x = torch.cat((H[st[0], st[1]], H[et[0], et[1]]), dim=-1)


        # these parts do not do anything: all node types in batches are either 1 or 2 and no virtuals
        if self.config['mean_pool'] : 
            is_special = torch.isin(input_ids, self.special_ids.to(input_ids.device))  # [B, S]
            combined_mask = att_mask * (~is_special).long()
            cH = mean_pooling(H, combined_mask)
            x[g.node_types == 0] = torch.cat((cH, cH), dim=-1)

        x[g.is_virtual == 1] = self.virtualH

        return x 
        

    def forward(self, g, input_ids, att_mask, st, et, anaphora_data=None) : 
        x  = self._get_input_embeddings(g, input_ids, att_mask, st, et)

        quote_x = x[g.node_types==1]#[g.is_pred]
        m_x = x[g.node_types==2]#[g.is_pred]
        
        
        embs = []
        ana_embs, ana_labels = [], []

        for bs in range(g.batch_q.max()+1) : 
            qx = quote_x[g.batch_q==bs][g.is_pred[bs]] #[q, H]
            mx = m_x[g.batch_m==bs]#[g.is_pred[bs]] # [m,H]]
            # scoring


            qx_rep = qx.repeat_interleave(mx.size(0), dim=0)   # [q*m, H]

            mx_rep = mx.repeat(qx.size(0), 1)                  # [q*m, H]

   
            # concatenate along feature dim
            out = torch.cat([qx_rep, mx_rep], dim=-1) 
            embs.append(out)

        embs = torch.cat(embs)
        scores = self.proj(embs).squeeze(-1)

        out = self._rearange(g, scores)

        out = {'qa_scores': out}
        
        return out
        
    def _loss(self, g, quote_x, m_x) : 
        pass
    
    def _rearange(self, g, scores) : 
        """Converts into list of [num_quotes, num_mentions] for each context window in the batch"""
        cnt = 0 
        outs = [] 
        for bs in range(g.batch_q.max()+1) :
            sm = (g.batch_m==bs).sum()
            sq = len(g.is_pred[bs])
            size = sq * sm
            outs.append(scores[cnt:cnt+size].view(sq, sm))
            cnt+=size

        return outs
    

