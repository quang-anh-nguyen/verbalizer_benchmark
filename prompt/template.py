import logging
import inspect

from openprompt.prompts import ManualTemplate

from utils import get_logger

logger = get_logger(__name__)

def get_template_cls_kwargs(template_type):
    if template_type=='manual':
        template_cls = ManualTemplate
    else:
        raise ValueError('Unimplemented template type')

    template_kwargs = list(inspect.signature(template_cls.__init__).parameters.keys())
    return template_cls, template_kwargs

def get_template(template_type, **kwargs):
    template_cls, template_kwargs = get_template_cls_kwargs(template_type)
    return template_cls(**{kw: kwargs[kw] for kw in kwargs if kw in template_kwargs})