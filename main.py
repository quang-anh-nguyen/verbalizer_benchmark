import argparse, logging
import json, os, random, string, shutil, glob
import torch
import time
import sys, shutil
import pandas as pd
import numpy as np

from data.data import get_dataset_processor

from pipeline.modeling import prepare_plm, PromptModelForClassification
from pipeline.trainer import get_training_arguments, get_trainer

from prompt.verbalizer import get_verbalizer
from prompt.template import get_template

from utils import get_logger, init_dl_program, postprocess_prediction, remove_logger, postprocess_history

torch.set_printoptions(sci_mode=False)


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

parser.add_argument('--verbalizer_type', type=str, choices=['manual', 'soft', 'auto', 'stat', 'ot', 'aug', 'augauto'], default='manual')
parser.add_argument('--verbalizer_id', type=int)
parser.add_argument('--verbalizer_file', type=str)

parser.add_argument('--num_labelword', type=int)
parser.add_argument('--augmented_num', type=int)

parser.add_argument('--ground_vocab', type=str)
parser.add_argument('--embedding_path', type=str)
parser.add_argument('--ground_dim', type=int, default=100)
parser.add_argument('--stat_distance', type=str)
parser.add_argument('--temperature', type=float, default=1., help='temperature for softmax of label logits')
parser.add_argument('--log_alpha', type=int, default=1)
parser.add_argument('--init_prototypes', action='store_true')
parser.add_argument('--calibrate', action='store_true')

parser.add_argument('--batchsize_train', type=int, default=4)
parser.add_argument('--batchsize_eval', type=int, default=4)
parser.add_argument('--truncate_test', type=int, default=0)

parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--max_steps', type=int, default=-1)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--freeze_lm', action='store_true')

parser.add_argument('--second_learning_rate', type=float, default=1e-3)

parser.add_argument('--valid_every_steps', type=float)
parser.add_argument('--valid_every_epochs', type=float)

parser.add_argument('--output_dir', type=str, default='./outputs')
parser.add_argument('--save_verbalizer_only', action='store_true')

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

logger.info(json.dumps(args.__dict__, indent=4))

logger.info(f'EXPERIMENT ID = {ID}')
print(f'EXPERIMENT ID = {ID}')
logger.info(f'Results saving at {OUTPUT_DIR}')
logger.debug(logging.Logger.manager.loggerDict)

logger.info(json.dumps(args.__dict__, indent=4))

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

if args.verbalizer_type in ['manual', 'soft', 'aug']:
    if args.verbalizer_id is None and args.verbalizer_file is None:
        args.verbalizer_id = 0

if not args.do_train:
    args.train_size=0

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
    do_stratify=True,
)

logger.info('Processor')
logger.info(processor)
logger.info('processor.template'+'='*100)
logger.info(processor.template)

# logger.info('Distribution of train')
# logger.info(np.unique(datasets['train'][processor.output], return_counts=True))

# logger.info('Distribution of valid')
# logger.info(np.unique(datasets['valid'][processor.output], return_counts=True))

if args.truncate_test:
    datasets['test'] = datasets['test'].select(range(args.truncate_test))

init_dl_program(args.device, seed=args.seed)

plm, tokenizer, plm_config, plm_wrapper = prepare_plm(args.model_type, args.model_path)

template = get_template(
    args.template_type,
    tokenizer=tokenizer,
    text=processor.template
)

if args.ground_vocab is not None:
    if args.ground_vocab=='tiny':
        ground = [w for v in processor.verbalizers for l in v for w in l]
        ground_construction = None
    elif 'json' in args.ground_vocab:
        with open(args.ground_vocab, 'r') as f:
            ground = json.load(f)
        ground_construction = None
    else:
        ground = None
else:
    ground = None
    ground_construction = None

verbalizer = get_verbalizer(
    args.verbalizer_type, 
    tokenizer=tokenizer, 
    model=plm, 
    label_words=processor.labelwords,
    classes=processor.labels,
    num_classes=len(processor.labels),
    label_word_num_per_class=args.num_labelword,
    augmented_num=args.augmented_num if args.verbalizer_type=='augauto' else args.num_labelword,
    ground=ground,
    ground_construction=ground_construction,
    loss_fn=args.stat_distance,
    temperature=args.temperature,
    log_alpha=bool(args.log_alpha),
    init_prototypes=args.init_prototypes,
    calibrate=args.calibrate,
    embedding_path=args.embedding_path,
    ground_dim=args.ground_dim,
    embeddings=plm.get_input_embeddings()
)

logger.info("="*100)
logger.info(type(verbalizer))
try:
    logger.info(verbalizer)
except:
    pass
logger.info('verbalizer')
# logger.info(processor.labelwords)
# logger.info(verbalizer.label_words)
# logger.info(verbalizer.label_words_ids)
logger.info("="*100)

prompt_model = PromptModelForClassification(plm, template, verbalizer, freeze_plm=args.freeze_lm)

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
    datasets=datasets if args.do_train else {},
    prompter=prompter,
)

logger.info("Trainer created")

results = {split: {} for split in datasets}

if args.verbalizer_type=='auto':
    args.do_zeroshot=False
if args.do_zeroshot:
    logger.info('Doing zeroshot')
    raw = trainer.predict(datasets['test'])
    results['test']['zero'] = postprocess_prediction(raw, processor.metrics)
    # logger.info(raw)

if args.do_train:

    trainer.train()

    log_history = trainer.state.log_history
    
    results.update(postprocess_history(log_history, processor.metrics))
    
    if not args.save_verbalizer_only:
        trainer.save_model(os.path.join(OUTPUT_DIR, 'checkpoint'))
    else:
        os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoint'))
        torch.save(verbalizer.state_dict(), os.path.join(OUTPUT_DIR, 'checkpoint', 'verbalizer.bin'))
        
    if os.path.isdir(os.path.join(OUTPUT_DIR, 'runs')):
        shutil.rmtree(os.path.join(OUTPUT_DIR, 'runs'))

if args.do_test:
    logger.info("Testing")
    raw = trainer.predict(datasets['test'])
    results['test']['best'] = postprocess_prediction(raw, processor.metrics)
    logger.info(json.dumps(results['test']['best']['metrics'], indent=4))

if (args.verbalizer_type=='auto') or (args.verbalizer_type=='augauto'):
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

with open(os.path.join(OUTPUT_DIR, 'verbalizer.json'), 'w+') as f:
    if args.verbalizer_type=='aug':
        json.dump({
            'ids': verbalizer.label_words_ids.detach().cpu().tolist(),
            'mask': verbalizer.words_ids_mask.detach().cpu().tolist(),
            'weight': verbalizer.label_words_weight.detach().cpu().tolist()
        }, f, indent=4)
    if args.verbalizer_type=='auto':
        json.dump({
            'ids': verbalizer.label_words_ids.detach().cpu().tolist(),
        }, f, indent=4)
    if args.verbalizer_type=='augauto':
        json.dump({
            'label_words': verbalizer.label_words,
            'ids': verbalizer.label_words_ids.detach().cpu().tolist(),
            'mask': verbalizer.words_ids_mask.detach().cpu().tolist(),
            'weight': verbalizer.label_words_weight.detach().cpu().tolist()
        }, f, indent=4)

logger.info(f"Verbalizer saved")


os.remove(os.path.join(OUTPUT_DIR, 'logfile.log'))
for ckpt in glob.glob(os.path.join(OUTPUT_DIR, 'checkpoint*')):
    print(ckpt)
    shutil.rmtree(ckpt)
