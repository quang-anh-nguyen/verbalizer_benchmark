import logging
import inspect
import torch

from openprompt.prompts import ManualVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from torch.nn import functional as F

from utils import get_logger

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

logger = get_logger(__name__)

def get_verbalizer_cls_kwargs(verbalizer_type):
    if verbalizer_type=='manual':
        verbalizer_cls = ManualVerbalizer
    elif verbalizer_type=='soft':
        verbalizer_cls = SoftVerbalizer
    elif verbalizer_type=='auto':
        verbalizer_cls = CustomAutomaticVerbalizer
    else:
        raise ValueError('Unimplemented verbalizer type')

    verbalizer_kwargs = list(inspect.signature(verbalizer_cls.__init__).parameters.keys())
    return verbalizer_cls, verbalizer_kwargs

def get_verbalizer(verbalizer_type, **kwargs):
    verbalizer_cls, verbalizer_kwargs = get_verbalizer_cls_kwargs(verbalizer_type)
    return verbalizer_cls(**{kw: kwargs[kw] for kw in kwargs if kw in verbalizer_kwargs})