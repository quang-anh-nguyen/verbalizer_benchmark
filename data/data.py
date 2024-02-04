import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from datasets import load_dataset, DatasetDict, ClassLabel

from data.processor import get_processor, has_multiple_verbalizers

def f1_macro(labels, preds):
    return f1_score(labels, preds, average='macro')

METRICS = {
    'acc': accuracy_score,
    'f1': f1_score,
    'matcor': matthews_corrcoef,
    'f1a': f1_macro
}

def get_metric(name):
    return METRICS[name]

def get_dataset_processor(
    dataset_name, 
    seed0=None, 
    seed1=None,
    train_size=32,
    train_to_valid=1.0,
    do_stratify=False,
    stratify_column=None,
    do_split=True,
    **kwargs
):

    if dataset_name in ['ag_news', 'yahoo', 'imdb', 'dbpedia', 'yelp', 'snli']:
        from_benchmark = False
        if dataset_name=='yahoo':
            original_dataset = load_dataset('yahoo_answers_topics')
        elif dataset_name=='dbpedia':
            original_dataset = load_dataset('dbpedia_14')
        elif dataset_name=='yelp':
            original_dataset = load_dataset('yelp_review_full')
        else:
            original_dataset = load_dataset(dataset_name)
    elif dataset_name in ['sst2', 'mnli', 'qnli', 'rte', 'mrpc', 'qqp', 'cola']:
        from_benchmark = True
        original_dataset = load_dataset('glue', dataset_name)
    elif dataset_name in ['sst5']:
        from_benchmark = False
        original_dataset = load_dataset('SetFit/sst5')
    elif 'factiva' in dataset_name:
        version = dataset_name.split('.')[-1]
        path = os.path.join('factiva', 'storage', version)
        from_benchmark = False
        if ('val' in version) or ('test' in version):
            original_dataset = load_dataset('json', data_files={'test': path+'/test.json'}, field='data')
        else:
            original_dataset = load_dataset('json', data_files={'train': path+'/train.json', 'test': path+'/test.json'}, field='data')
        original_dataset = original_dataset.class_encode_column('sector')
    elif dataset_name=='mlsum.fr':
        from_benchmark = False
        original_dataset = load_dataset('mlsum', 'fr')
        
        for split in original_dataset:
            i = pd.read_parquet(f"mlsum_fr/{split}")
            original_dataset[split] = original_dataset[split].select(i['index'].tolist())
            original_dataset[split] = original_dataset[split].add_column('label', i['label'].tolist())
            
        original_dataset = original_dataset.cast_column('label', ClassLabel(names=['Economie', 'Opinion', 'Politique', 'Societe', 'Culture', 'Sport', 'Environement', 'Technologie', 'Education', 'Justice']))
    else:
        raise Exception("Dataset processor not implemented yet")

    if 'factiva' not in dataset_name:
        if has_multiple_verbalizers(dataset_name):
            processor = get_processor(dataset_name)(**{kw: kwargs[kw] for kw in kwargs if kw in ['template_id', 'verbalizer_id', 'verbalizer_file']})
        else:
            assert kwargs['verbalizer_id']==0, f"Get verbalizer_id = {kwargs['verbalizer_id']} for only one default verbalizer."
            processor = get_processor(dataset_name)(**{kw: kwargs[kw] for kw in kwargs if kw in ['template_id', 'verbalizer_id', 'verbalizer_file']})
    else:
        processor = get_processor('factiva')(
            classes=original_dataset['test'].features['sector'].names, 
            **{kw: kwargs[kw] for kw in kwargs if kw in ['template_id', 'verbalizer_id', 'verbalizer_file']}
        )

    if train_size<=0:
        do_split = False
    if not do_split:
        return original_dataset, processor
    
    if do_stratify and stratify_column is None:
        stratify_column = processor.output

    if len(original_dataset['train'])==int(train_size*(1+train_to_valid)):        
        label_data = original_dataset['train'].train_test_split(
            train_size=(train_to_valid)/(train_to_valid+1),
            stratify_by_column=stratify_column,
            seed=seed1
        )
    else:
        label_data = original_dataset['train'].train_test_split(
            train_size=int(train_size*(1+train_to_valid)), 
            stratify_by_column=stratify_column,
            seed=seed0)
    
        label_data = label_data['train'].train_test_split(
            train_size=(train_to_valid)/(train_to_valid+1),
            stratify_by_column=stratify_column,
            seed=seed1
        )
    
    datasets = DatasetDict({
        'train': label_data['train'],
        'valid': label_data['test'],
        'test': original_dataset['valid' if from_benchmark else 'test']
    })
    
    return datasets, processor
