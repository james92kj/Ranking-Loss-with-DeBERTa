import torch 
import torch.nn as nn 
from transformers import AutoConfig, AutoModel


def get_ranking_loss(logits, labels, margin=0.7):
    logits = torch.sigmoid(logits)

    # labels [1,n], [n,1]
    labels1 = labels.unsqueeze(0)
    labels2 = labels.unsqueeze(1)

    # logits [1,n], [n,1]
    logits1 = logits.unsqueeze(0)
    logits2 = logits.unsqueeze(1)
    
    # calculate the label differences to track the ordering 
    y_ij = torch.sign(labels1 - labels2)
    p_ij = logits1 - logits2

    loss = torch.clamp(-p_ij * y_ij + margin,min=0.0).mean()
    return loss


class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling,self).__init__() 

    def forward(self, hidden_act, attention_mask):
        t_attention_mask = attention_mask.unsqueeze(-1).expand(hidden_act.shape).float()
        embeddings_for_seq = torch.sum(hidden_act * t_attention_mask,dim=1)
        sum_mask = t_attention_mask.sum(1)
        t_attention_mask = torch.clamp(t_attention_mask, min=1e-9)
        mean_embeddings = embeddings_for_seq/sum_mask
        return mean_embeddings


class AiModel(nn.Module):

    def __init__(self, cfg):
        super(AiModel,self).__init__()
        
        backbone_config = AutoConfig.from_pretrained(cfg.model.backbone_path)

        # disables the KV cache during training 
        backbone_config.update({
            'use_cache':False
        })
        
        self.backbone = AutoModel.from_pretrained(cfg.model.backbone_path, config=backbone_config)

        # enable gradient checkpointing to save memory. 
        if cfg.model.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        hidden_size = backbone_config.hidden_size
        self.dropout = nn.Dropout(cfg.model.dropout_rate)
        self.scorer = nn.Linear(hidden_size,1)
        self.pool = MeanPooling()
        

    def forward(self,input_ids, attention_mask, labels=None):

        backbone_output = self.backbone(input_ids,attention_mask) 
        last_hidden_state = backbone_output.last_hidden_state

        output = self.pool(last_hidden_state, attention_mask)
        output = self.dropout(output)
        scores = self.scorer(output)
        scores = scores.reshape(-1)

        loss = None
        if labels is not None:
            labels = labels.reshape(-1)
            loss = get_ranking_loss(scores,labels=labels)

        return scores, loss 








