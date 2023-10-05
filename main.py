import argparse, logging
import json, os, random, string
import torch
import time
import sys
import pandas as pd

from data.data import get_dataset_processor

from pipeline.modeling import prepare_plm, PromptModelForClassification
from pipeline.trainer import get_training_arguments, get_trainer

from prompt.verbalizer import get_verbalizer
from prompt.template import get_template

from utils import get_logger, init_dl_program, postprocess_prediction, remove_logger, postprocess_history


parser = argparse.ArgumentParser()

parser.add_argument('model_type', type=str, help='LM family name')
parser.add_argument('model_path', type=str, help='path to load pretrained LM')

parser.add_argument('dataset', type=str, help='dataset path')

parser.add_argument('--do_zeroshot', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_valid', action='store_true')
parser.add_argument('--do_test', action='store_true')
parser.add_argument('--experiment_id', type=str)
parser.add_argument('--experiment_name', type=str, default='')

parser.add_argument('--seed', type=int, help='seed for init_dl_program')

parser.add_argument('--train_size', type=int, help='total number of labeled data for train+valid', default=32)
parser.add_argument('--train_to_valid', type=int, help='train/valid ratio', default=1)
parser.add_argument('--split', type=int, help='seed for data spliting', nargs=2, default=[None, None])

parser.add_argument('--template_type', choices=['manual', 'prefix', 'ptuning', 'mixed'], default='manual')
parser.add_argument('--template_id', type=int)
parser.add_argument('--template_file', type=str)

parser.add_argument('--verbalizer_type', type=str, choices=['manual', 'soft', 'auto'], default='manual')
parser.add_argument('--verbalizer_id', type=int)
parser.add_argument('--verbalizer_file', type=str)
parser.add_argument('--num_labelword', type=int)

parser.add_argument('--batchsize_train', type=int, default=4)
parser.add_argument('--batchsize_eval', type=int, default=4)
parser.add_argument('--truncate_test', type=int, default=0)

parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--max_steps', type=int)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--second_learning_rate', type=float, default=1e-3)

parser.add_argument('--valid_every_steps', type=float)
parser.add_argument('--valid_every_epochs', type=float)

parser.add_argument('--output_dir', type=str, default='./outputs')

parser.add_argument('--device', choices=['cuda:0', 'cpu'], type=str, default='cpu')

# parser.add_argument('')

args = parser.parse_args()

if args.experiment_id is None:
    ID = random.choices(string.digits, k=10)
    ID = ''.join(ID)
    args.experiment_id = ID
else:
    ID = args.experiment_id

OUTPUT_DIR = os.path.join(args.output_dir, args.dataset, args.experiment_name, ID)  
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logger = get_logger(__name__, filename=os.path.join(OUTPUT_DIR, 'logfile.log'))
get_logger('datasets', filename=os.path.join(OUTPUT_DIR, 'logfile.log'))
get_logger('transformers', filename=os.path.join(OUTPUT_DIR, 'logfile.log'))

# for n in (logging.Logger.manager.loggerDict):
#     print(n)
# raise Exception()

logger.info(args.__dict__)

logger.info(f'EXPERIMENT ID = {ID}')
logger.info(f'Results saving at {OUTPUT_DIR}')
logger.debug(logging.Logger.manager.loggerDict)

# raise Exception()

if torch.cuda.is_available():
    pass
else:
    logger.warning('Cuda unavailable')
    args.device = 'cpu'
logger.info(f'Using device: {args.device}')

if args.template_type in ['manual']:
    if args.template_id is None and args.template_file is None:
        args.template_id = 0

logger.info('args.template_id'+'='*100)
logger.info(args.template_id)

if args.verbalizer_type in ['manual', 'soft']:
    if args.verbalizer_id is None and args.verbalizer_file is None:
        args.verbalizer_id = 0

datasets, processor = get_dataset_processor(
    args.dataset, 
    seed0=args.split[0], 
    seed1=args.split[1], 
    train_size=args.train_size, 
    train_to_valid=args.train_to_valid,
    template_id=args.template_id, 
    template_file=args.template_file,
    verbalizer_id=args.verbalizer_id,
    verbalizer_file=args.verbalizer_file,
)

logger.info('processor.template'+'='*100)
logger.info(processor.template)

if args.truncate_test:
    datasets['test'] = datasets['test'].select(range(args.truncate_test))

init_dl_program(args.device, seed=args.seed)

plm, tokenizer, plm_config, plm_wrapper = prepare_plm(args.model_type, args.model_path)

template = get_template(
    args.template_type,
    tokenizer=tokenizer,
    text=processor.template
)

print('='*10)
print(processor.labelwords)
print('='*10)

if False:
    pass
else:
    verbalizer = get_verbalizer(
        args.verbalizer_type, 
        tokenizer=tokenizer, 
        model=plm, 
        label_words=processor.labelwords,
        classes=processor.labels,
        num_classes=len(processor.labels),
        label_word_num_per_class=args.num_labelword,
    )

prompt_model = PromptModelForClassification(plm, template, verbalizer)

prompter = {
    'processor': processor,
    'template': template,
    'verbalizer': verbalizer,
    'plm_wrapper': plm_wrapper,
    'tokenizer': tokenizer
}

training_args = get_training_arguments(args)

trainer = get_trainer(
    args, 
    model=prompt_model,
    args=training_args,
    datasets=datasets,
    prompter=prompter,
)

results = {split: {} for split in datasets}

if args.verbalizer_type=='auto':
    args.do_zeroshot=False
if args.do_zeroshot:
    raw = trainer.predict(datasets['test'])
    results['test']['zero'] = postprocess_prediction(raw, processor.metrics)

if args.do_train:

    trainer.train()

    log_history = trainer.state.log_history
    
    results.update(postprocess_history(log_history, processor.metrics))
    
    trainer.save_model(os.path.join(OUTPUT_DIR, 'checkpoint'))

if args.do_test:
    raw = trainer.predict(datasets['test'])
    results['test']['best'] = postprocess_prediction(raw, processor.metrics)

if args.verbalizer_type=='auto':
    pass
    tokens = {processor.labels[i]: tokenizer.convert_ids_to_tokens(w, skip_special_tokens=True) for i, w in enumerate(verbalizer.label_words_ids)}
    # tokens = {c: [w[1:] if w[0]=='Ġ' else w for w in tokens[c]] for c in tokens}
    with open(os.path.join(OUTPUT_DIR, 'verbalizer.json'), 'w+', encoding='utf8') as f:
        json.dump(tokens, f, indent=4, ensure_ascii=False)
    logger.info(f'Learned verbalizer saved')
    logger.info(json.dumps(tokens, indent=4))

with open(os.path.join(OUTPUT_DIR, 'info.json'), 'w+') as f:
    json.dump({
        'id': ID,
        'args': args.__dict__,
        'results': results
    }, f, indent=4)
logger.info(f'Result saved to {OUTPUT_DIR}')
          

