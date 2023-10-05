from datasets import Dataset, Sequence, Value
from typing import List, Dict, Any

from openprompt.pipeline_base import PromptDataLoader
from data.processor import process_dataset

def get_prompt_dataloader(processor, dataset, template, tokenizer, plm_wrapper=None, batch_size=1, shuffle=False):
    return PromptDataLoader(
        dataset=process_dataset(processor, dataset), 
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=plm_wrapper,
        batch_size=batch_size,
        shuffle=shuffle,
        decoder_max_length=tokenizer.model_max_length
    )

def transpose(l: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {k: ([r[k].tolist() for r in l]) for k in l[0]}
