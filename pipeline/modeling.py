import numpy as np
import torch
import os, logging

from openprompt.pipeline_base import PromptForClassification
# from torch.nn import CrossEntropyLoss
from openprompt.plms import load_plm
from openprompt.plms.mlm import MLMTokenizerWrapper
from tqdm import tqdm
# from transformers import get_linear_schedule_with_warmup
# from torch.optim import AdamW

from data.data import METRICS

LOSS_DIVIDE = 100

def prepare_plm(model_type, model_path):
    try:
        plm, tokenizer, config, wrapper_plm = load_plm(model_type, model_path)
    except:
        if model_type=='deberta':
            from transformers import DebertaConfig, DebertaTokenizer, DebertaForMaskedLM
            plm = DebertaForMaskedLM.from_pretrained(model_path)
            tokenizer = DebertaTokenizer.from_pretrained(model_path)
            config = DebertaConfig.from_pretrained(model_path)
            wrapper_plm = MLMTokenizerWrapper
        if model_type=='camembert':
            from transformers import CamembertConfig, CamembertTokenizer, CamembertForMaskedLM
            plm = CamembertForMaskedLM.from_pretrained(model_path)
            tokenizer = CamembertTokenizer.from_pretrained(model_path)
            config = CamembertConfig.from_pretrained(model_path)
            wrapper_plm = MLMTokenizerWrapper
        elif model_type in ['gottbert', 'roberta-bne', 'ruberta']:
            from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM
            plm = RobertaForMaskedLM.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
            config = RobertaConfig.from_pretrained(model_path)
            wrapper_plm = MLMTokenizerWrapper
    return plm, tokenizer, config, wrapper_plm

class PromptModelForClassification(PromptForClassification):
    def __init__(
            self,
            plm,
            template,
            verbalizer,
            freeze_plm=False,
            plm_eval_mode=False
    ):
        super().__init__(plm, template, verbalizer, freeze_plm, plm_eval_mode)
    
    def forward(
            self, 
            input_ids=None,
            token_type_ids=None,
            attention_mask=None, 
            loss_ids=None, 
            labels=None,
            soft_token_ids=None,
            past_key_values=None,
            ):
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None:
            batch['token_type_ids'] = token_type_ids
        if loss_ids is not None:
            batch['loss_ids'] = loss_ids
        if labels is not None:
            batch['label'] = labels
        if soft_token_ids is not None:
            batch['soft_token_ids'] = soft_token_ids
        if past_key_values is not None:
            batch['past_key_values'] = past_key_values

        # outputs = self.prompt_model(batch)
        # outputs = {
        #     'logits': self.extract_at_mask(outputs.logits, batch),
        #     'last_hidden_state': self.extract_at_mask(outputs.hidden_states[-1], batch)
        # }
        # outputs = self.extract_at_mask(outputs, batch)
        logits = super().forward(batch)
        return {'logits': logits}
        
