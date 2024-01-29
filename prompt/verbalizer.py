import logging
import inspect
import torch
import numpy as np
import json

from typing import Optional, Sequence, Union
from transformers.tokenization_utils import PreTrainedTokenizer

from openprompt.prompts import ManualVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt import Verbalizer
from torch import nn as nn
from torch.nn import functional as F

from utils import get_logger
from tqdm import tqdm

class CustomAutomaticVerbalizer(AutomaticVerbalizer):

    def register_buffer(self, logits, labels):
        r'''

        Args:
            logits (:obj:`torch.Tensor`):
            labels (:obj:`List`):
        '''

        logits = F.softmax(logits.detach(),dim=-1).cpu()
        labels = labels.detach().cpu()
        if self.probs_buffer is None :
            self.probs_buffer = logits
            self.labels_buffer = labels
        else:
            self.probs_buffer = torch.vstack([self.probs_buffer, logits])
            self.labels_buffer = torch.hstack([self.labels_buffer, labels])


            
def entropy(p: torch.Tensor) -> torch.Tensor:
    return - (p * torch.log(p)).sum(-1)

class ScalarProduct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return - torch.matmul(a, b.T)
    
class TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return (a[:, None, :] - b[None, :, :]).abs().sum(-1)/2
    
class HellingerDistance(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        a = torch.sqrt(a)
        b = torch.sqrt(b)
        return ((a[:, None, :] - b[None, :, :])**2).sum(-1)/2
    
class KullbackLeiblerDivergence(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return torch.sum(a[:, None, :] * (torch.log(a[:, None, :]) - torch.log(b[None, :, :])), -1)

class JensenShannonDivergence(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        m = (a[:, None, :] + b[None, :, :])/2
        return (entropy(m) - entropy(a)[:, None]/2 - entropy(b)[None, :]/2)/np.log(2)

STATISTICAL_DISTANCES = {
    'dot': ScalarProduct,
    'tv': TotalVariation,
    'hl': HellingerDistance,
    'kl': KullbackLeiblerDivergence,
    'js': JensenShannonDivergence
}

class AugmentedVerbalizer(ManualVerbalizer):
    
    def __init__(
        self,
        tokenizer=None,
        classes=None,
        num_classes=None,
        label_words=None,
        post_log_softmax=True,
        prefix=' ',
        augmented_num=0,
        embeddings=None,
    ):
    
        super().__init__(
            tokenizer=tokenizer,
            classes=classes,
            num_classes=num_classes,
            label_words=None,
            post_log_softmax=post_log_softmax,
            prefix=prefix,
        )
        
        self.augmented_num = augmented_num
        self.embeddings = embeddings
        self.label_words = label_words
        
    def generate_parameters(self):
        
        logger.info("Generating parameters for: "+json.dumps(self.label_words))
              
        all_ids = []
        all_weights = []
        for words_per_label in self.label_words:
            ids_per_label = []
            weights_per_label = [1.0] * len(words_per_label)
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)            
                
                base_embedding = self.embeddings(torch.tensor(ids)).mean(0)
                candidate_embeddings = self.embeddings(torch.tensor([i for i in range(self.tokenizer.vocab_size)]))
                similarities = F.cosine_similarity(base_embedding[None, :], candidate_embeddings, dim=-1)
                values, idx = torch.topk(similarities, self.augmented_num+1)
                
                values = values[1:]
                idx = idx[1:]
                
                logger.info(f"{word} {ids} choose {self.tokenizer.convert_ids_to_tokens(idx)}")

                ids_per_label += [[i] for i in idx]
                weights_per_label += values
            
            all_ids.append(ids_per_label)
            all_weights.append(weights_per_label)
        
        to_show = [[self.tokenizer.decode(j) for j in i] for i in all_ids]
        logger.info("Verbalizer using: \n"+json.dumps(to_show, indent=4))
                    
        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])

        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]
        
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]
        
        weights = [weights_per_label + [0]*(max_num_label_words-len(weights_per_label)) for weights_per_label in all_weights]
        weights = torch.tensor(weights)

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        self.label_words_weight = nn.Parameter(weights, requires_grad=True)
        
        logger.info(self.label_words_ids.shape)
        logger.info(self.words_ids_mask.shape)
        logger.info(self.label_words_mask.shape)
        logger.info(self.label_words_weight.shape)
        logger.info(weights)
        
    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000*(1-self.label_words_mask)
        return label_words_logits
    
    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
       
        label_words_logits = (label_words_logits * self.label_words_mask * self.label_words_weight).sum(-1)/(self.label_words_weight*self.label_words_mask).sum(-1)
        return label_words_logits
    

class StatisticalVerbalizer(Verbalizer):

    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 classes: Optional[Sequence[str]] = None,
                 num_classes: Optional[int] = None,
                 ground: Optional[Sequence[Union[str, int]]] = None,
                 ground_construction: Optional[str] = None,
                 loss_fn: Optional[str] = 'dot',
                 post_log_softmax: Optional[bool] = True,
                 log_alpha=True,
                 temperature=1.,
                 prefix=' ',
                 init_by_train=False,
                 calibrate=True
                 ):
        
        super().__init__(tokenizer, classes, num_classes)
        if ground is None and ground_construction is None:
            raise ValueError("Both ground and ground construction can not be None.")
        if ground is not None:
            if isinstance(ground[0], str):
                ground_ids = []
                for word in ground:
                    ground_ids.append(tokenizer.convert_tokens_to_ids(word))
            else:
                ground_ids = ground
            self.ground_ids = ground_ids
        else:
            if ground_construction=='all':
                self.ground_ids = list(range(self.tokenizer.vocab_size))
            else:
                self.ground_ids = None
        
        self.ground_dim = len(self.ground_ids)
        self.prefix = prefix

        self.loss_fn = STATISTICAL_DISTANCES[loss_fn]()
        self.post_log_sofmax = post_log_softmax

        self.log_alpha = log_alpha
        self.temperature = temperature
            
        self.labels_buffer = None
        self.logits_buffer = None
        
        self.calibrate = calibrate
        self.ground_prior = nn.Parameter(torch.zeros(self.ground_dim), requires_grad=False)
        
        if ground is None:
            self.keep_vocab_logits = True
        else:
            self.keep_vocab_logits = False
        
        if self.init_prototypes:
            self.keep_word_logits = True
        else:
            self.keep_word_logits = False

        self.generate_parameters()

    def register_buffers(self, labels, logits):
        if self.labels_buffer is None:
            self.labels_buffer = torch.tensor(labels)
            self.logits_buffer = torch.tensor(logits)
        else:
            self.labels_buffer = torch.vstack([self.labels_buffer, labels])
            self.logits_buffer = torch.vstack([self.logits_buffer, logits])
    
    def init_prototypes(self, labels: torch.Tensor=None, logits: torch.Tensor=None):
        if labels is None:
            labels = self.labels_buffer
        if logits is None:
            logits = self.logits_buffer
            
        # logger.info(labels.cpu().tolist())
        # logger.info(torch.round(logits, decimals=2).cpu().to(torch.float32))
        for l in range(self.num_classes):
            if self.log_alpha:
                class_support = logits[labels==l].to(self.prototypes.data)  -self.ground_prior[None, :]
            else:
                class_support = F.softmax(logits[labels==l].to(self.prototypes.data) -self.ground_prior[None, :], -1)
            
            self.prototypes.data[l] = class_support.mean(0)
            
        self.prototypes = nn.Parameter(self.prototypes)
        self.labels_buffer = None
        self.logits_buffer = None
        self.keep_word_logits = False
        
        # logger.info('here')
        # logger.info(torch.round(self.prototypes.data, decimals=2).cpu().to(torch.float32))
        # raise Exception()
            

    def generate_parameters(self, ):
        if self.ground_ids is None:
            return            
            
        if self.log_alpha:
            self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.ground_dim), requires_grad=True)    
        else:
            self.prototypes = nn.Parameter(torch.ones(self.num_classes, self.ground_dim) / self.ground_dim, requires_grad=True)
        
        self.temperature = nn.Parameter(torch.tensor(self.temperature), requires_grad=False)
        if self.ground_ids is not None:
            self.ground_ids = nn.Parameter(torch.tensor(self.ground_ids), requires_grad=False)
            
        # if self.calibrate:
        #     self.ground_prior = nn.Parameter(self.ground_prior, requires_grad=False)


    def process_logits(self, logits: torch.Tensor, **kwargs):
        if self.keep_vocab_logits:
            return logits
        
        if self.ground_ids is None:
            self.register_buffers(logits, kwargs['batch']['label'])
        ground_logits = self.project(logits, **kwargs)
        if self.post_log_sofmax:
            ground_probas = self.normalize(ground_logits, **kwargs)
            ground_logits = torch.log(ground_probas + 1e-20) - self.ground_prior[None, :]
        
        if self.keep_word_logits:
            return ground_logits
        
        class_logits = self.aggregate(ground_logits, **kwargs)
        return class_logits
        
    def project(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return logits[:, self.ground_ids]
    
    def normalize(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.softmax(logits, dim=-1)
    
    def aggregate(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(self.loss_fn, ScalarProduct):
            samples = logits
        else:
            samples = F.softmax(logits, -1)
        if self.log_alpha:
            prototypes = F.softmax(self.prototypes, -1)
        else:
            prototypes = self.prototypes/self.prototypes.sum(-1, keepdim=True)
        results = - self.loss_fn(samples, prototypes)
        return results/self.temperature
    
from ddn.pytorch.optimal_transport import sinkhorn, OptimalTransportLayer
    
class OptimalTransportVerbalizer(Verbalizer):
    
    def __init__(
        self,         
        tokenizer: Optional[PreTrainedTokenizer] = None,
        classes: Optional[Sequence[str]] = None,
        num_classes: Optional[int] = None,
        post_log_softmax=True,
        prefix=' ',
        ground_dim=-1,
        init_by_train=True,
        calibrate=True,
        embedding_path='',
        freeze_embeddings=True,
        underlying_metric='cos'
        ):
        
        super().__init__(tokenizer, classes, num_classes)
        
        self.post_log_sofmax = post_log_softmax
        self.prefix = prefix
        
        self.freeze_embeddings = freeze_embeddings
        self.embedding_path = embedding_path
        self.underlying_metric = underlying_metric
                              
        self.optimal_transport = OptimalTransportLayer()
        self.calibrate = calibrate
        
        self.ground_dim = ground_dim
        self.ground_ids = None
        self.embeddings = None
        
        self.ground_prior = None
        
        self.labels_buffer = None
        self.logits_buffer = None
        
        self.init_by_train = init_by_train
        self.initialized = False
        
        # self.generate_parameters()
        
    def filter_ground(self, k):
        
        candidate_logits = self.ground_prior[self.ground_ids]
        idx = torch.argsort(candidate_logits, descending=True)[:k]
        
        self.ground_ids = self.ground_ids[idx]
        self.embeddings = self.embeddings[idx]
        
        self.ground_prior = self.ground_prior[idx]
                        
    def generate_parameters(self):
        
        reader = torch.load(self.embedding_path)
        self.ground_ids = reader['indices'].cpu()
        self.embeddings = reader['vectors'].cpu()
        
        if self.ground_dim<=0:
            self.ground_dim = len(self.ground_ids)
        else:
            self.filter_ground(self.ground_dim)
            
        logger.info(f"Ground dimension is {self.ground_dim}")
    
        distances = []
        logger.info("Calculating similarities")
        if self.underlying_metric=='cos':
            for i in tqdm(range(self.ground_dim)):
                distances.append(F.cosine_similarity(self.embeddings.unsqueeze(0), self.embeddings, dim=-1))
        self.distance = torch.vstack(distances)
        
        self.ground_ids = nn.Parameter(self.ground_ids, requires_grad=False)
        self.embeddings = nn.Parameter(self.embeddings, requires_grad=not self.freeze_embeddings)
        
        if self.calibrate and (self.ground_prior is not None):
            self.ground_prior = nn.Parameter(self.ground_prior, requires_grad=False)
        else:
            self.ground_prior = nn.Parameter(torch.zeros(len(self.ground_prior)), requires_grad=False)
            
        self.init_prototypes()
        
    def register_buffers(self, labels, logits):
        if self.labels_buffer is None:
            self.labels_buffer = torch.tensor(labels)
            self.logits_buffer = torch.tensor(logits)
        else:
            self.labels_buffer = torch.cat([self.labels_buffer, labels], dim=0)
            self.logits_buffer = torch.cat([self.logits_buffer, logits], dim=0)
            
    def init_prototypes(self, labels: torch.Tensor=None, logits: torch.Tensor=None):
        
        self.prototypes = torch.zeros(self.num_classes, self.ground_dim)
        if self.init_by_train:
            if labels is None:
                labels = self.labels_buffer
            if logits is None:
                logits = self.logits_buffer
                
            logits = logits[:, self.ground_ids]

            for l in range(self.num_classes):
                class_support = logits[labels==l].to(self.prototypes.data) - self.ground_prior[None, :]
                self.prototypes.data[l] = class_support.mean(0)
                
        self.prototypes = nn.Parameter(self.prototypes)
        self.labels_buffer = None
        self.logits_buffer = None
        self.initialized = True
        
    def process_logits(self, logits: torch.Tensor, **kwargs):
        if self.ground_ids is None:
            self.register_buffers(kwargs['batch']['label'], logits)
            return logits
  
        ground_logits = self.project(logits, **kwargs)
        if self.post_log_sofmax:
            ground_probas = self.normalize(ground_logits, **kwargs)
            ground_logits = torch.log(ground_probas + 1e-20) - self.ground_prior[None, :]
        
        class_logits = self.aggregate(ground_logits, **kwargs)
        return class_logits
        
    def project(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return logits[:, self.ground_ids]
    
    def normalize(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def aggregate(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        probas = F.softmax(logits, -1)
        prototypes = F.softmax(self.prototypes, -1)
        
        B, D = probas.shape
        C, D = prototypes.shape
        
        M = self.distance.to(probas)
        OT_loss = torch.zeros(B, C).to(probas)
        for c in range(C):
            for b in range(B):
                P = (self.optimal_transport(M=M, r=prototypes[[c]], c=probas[[b]]))
            OT_loss[b, c] = (P*M).sum()
            
        return - OT_loss

logger = get_logger(__name__)

def get_verbalizer_cls_kwargs(verbalizer_type):
    if verbalizer_type=='manual':
        verbalizer_cls = ManualVerbalizer
    elif verbalizer_type=='soft':
        verbalizer_cls = SoftVerbalizer
    elif verbalizer_type=='auto':
        verbalizer_cls = CustomAutomaticVerbalizer
    elif verbalizer_type=='stat':
        verbalizer_cls = StatisticalVerbalizer
    elif verbalizer_type=='ot':
        verbalizer_cls = OptimalTransportVerbalizer
    elif verbalizer_type=='aug':
        verbalizer_cls = AugmentedVerbalizer
    else:
        raise ValueError('Unimplemented verbalizer type')

    verbalizer_kwargs = list(inspect.signature(verbalizer_cls.__init__).parameters.keys())
    return verbalizer_cls, verbalizer_kwargs

def get_verbalizer(verbalizer_type, **kwargs):
    verbalizer_cls, verbalizer_kwargs = get_verbalizer_cls_kwargs(verbalizer_type)
    return verbalizer_cls(**{kw: kwargs[kw] for kw in kwargs if kw in verbalizer_kwargs})