import argparse
import json
import os
import torch

from utils import get_logger, postprocess_prediction, init_dl_program
from data.data import get_dataset_processor
from pipeline.modeling import prepare_plm, PromptModelForClassification
from pipeline.trainer import get_trainer, get_evaluating_arguments
from prompt.template import get_template
from prompt.verbalizer import get_verbalizer

EPSILON = 1e-8

parser = argparse.ArgumentParser()

# parser.add_argument('model_type', type=str, help='LM family name')
# parser.add_argument('model_path', type=str, help='path to load pretrained LM')

# parser.add_argument('dataset', type=str, help='dataset path')

# parser.add_argument('--experiment_id', type=str)
# parser.add_argument('--experiment_name', type=str, default='')

# parser.add_argument('--seed', type=int, help='seed for init_dl_program')

parser.add_argument('dir')
parser.add_argument('--data_path', type=str)

# parser.add_argument('--template_type', choices=['manual', 'prefix', 'ptuning', 'mixed'], default='manual')
# parser.add_argument('--template_id', type=int)
# parser.add_argument('--template_file', type=str)

# parser.add_argument('--verbalizer_type', type=str, choices=['manual', 'soft', 'auto'], default='manual')
# parser.add_argument('--verbalizer_id', type=int)
# parser.add_argument('--verbalizer_file', type=str)
# parser.add_argument('--num_labelword', type=int, default=3)

# parser.add_argument('--batchsize_train', type=int, default=4)
# parser.add_argument('--batchsize_eval', type=int, default=4)
# parser.add_argument('--truncate_test', type=int, default=1000)

# parser.add_argument('--learning_rate', type=float, default=1e-5)
# parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
# parser.add_argument('--epochs', type=int, default=10)
# parser.add_argument('--max_steps', type=int)
# parser.add_argument('--warmup', type=int, default=0)
# parser.add_argument('--weight_decay', type=float, default=0)

# parser.add_argument('--second_learning_rate', type=float, default=1e-3)

# parser.add_argument('--valid_every_steps', type=float)
# parser.add_argument('--valid_every_epochs', type=float)

# parser.add_argument('--output_dir', type=str, default='./outputs')

# parser.add_argument('--device', choices=['cuda:0', 'cpu'], type=str, default='cpu')

logger = get_logger(__name__)
get_logger('datasets')
get_logger('transformers')

ARGS = parser.parse_args()
DIR = ARGS.dir
DATA_PATH = ARGS.data_path
another_data = DATA_PATH is not None

with open(os.path.join(DIR, 'info.json')) as f:
    info = json.load(f)

ID = info['id']
logger.info(f"EXPERIMENT ID: {ID}")

args = argparse.Namespace(**info['args'])
if args.experiment_id is None:
    args.experiment_id = ID

if torch.cuda.is_available():
    pass
else:
    logger.warning('Cuda unavailable')
    args.device = 'cpu'
logger.info(f'Using device: {args.device}')

if args.template_type in ['manual']:
    if args.template_id is None and args.template_file is None:
        args.template_id = 0

if args.verbalizer_type in ['manual', 'soft']:
    if args.verbalizer_id is None and args.verbalizer_file is None:
        args.verbalizer_id = 0
elif args.verbalizer_type in ['auto']:
    args.verbalizer_file = os.path.join(DIR, 'verbalizer.json')

# logger.info(args.verbalizer_file)
# input()

logger.info('Getting datasets')
if another_data:
    datasets, processor = get_dataset_processor(
        DATA_PATH, 
        train_size=args.train_size, 
        template_id=args.template_id, 
        template_file=args.template_file,
        verbalizer_id=args.verbalizer_id,
        verbalizer_file=args.verbalizer_file,
        do_split=False
    )
else:
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
        do_split=True
    )

init_dl_program(args.device, seed=args.seed)

plm, tokenizer, plm_config, plm_wrapper = prepare_plm(args.model_type, args.model_path)

logger.info('Creating template')
template = get_template(
    args.template_type,
    tokenizer=tokenizer,
    text=processor.template
)

logger.info('Creating verbalizer')
if args.verbalizer_type=='auto':
    verbalizer = get_verbalizer(
        'manual',
        tokenizer=tokenizer,
        model=plm,
        classes=processor.labels,
        label_words=processor.labelwords
    )
else:
    verbalizer = get_verbalizer(
        args.verbalizer_type, 
        tokenizer=tokenizer, 
        model=plm, 
        label_words=processor.labelwords,
        classes=processor.labels,
        num_classes=len(processor.labels),
        label_word_num_per_class=args.num_labelword if hasattr(args, 'num_labelword') else None,
    )

logger.info(f'Using labelwords:\n{processor.labelwords}')

logger.info('Creating model')
prompt_model = PromptModelForClassification(plm, template, verbalizer)

# with open(os.path.join(DIR, 'checkpoint', 'pytorch_model.bin')) as f:
    # checkpoint = torch.load(f)

logger.info('Prepare done')

prompter = {
    'processor': processor,
    'template': template,
    'verbalizer': verbalizer,
    'plm_wrapper': plm_wrapper,
    'tokenizer': tokenizer
}

training_args = get_evaluating_arguments(args)

trainer = get_trainer(
    args, 
    model=prompt_model,
    args=training_args,
    datasets=datasets,
    prompter=prompter,
)
logger.info('Trainer created')

result = {}

print(args.do_zeroshot)
if args.verbalizer_type=='auto':
    args.do_zeroshot = False
if args.do_zeroshot:
    logger.info('Doing zeroshot')
    raw = trainer.predict(datasets['test'])
    result['zero'] = postprocess_prediction(raw, processor.metrics)

logger.info('Loading checkpoint')
checkpoint = torch.load(os.path.join(DIR, 'checkpoint', 'pytorch_model.bin'))

missing, _ = prompt_model.load_state_dict(checkpoint, strict=False)
if len(missing)>0:
    assert all([x.startswith('verbalizer') for x in missing]), "Missing other parameters other than the verbalizer"
    assert args.verbalizer_type=='auto', "Missing verbalizer parameters but not automatic verbalizer"

    all_ids = [[[tokenizer.convert_tokens_to_ids(word)] for word in words_per_label] for words_per_label in processor.labelwords]
    max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
    assert max_len==1, "Label words are being split to more than one tokenizers, please check"
    max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
      
    words_ids_mask = torch.zeros(max_num_label_words, max_len)
    words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label] + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) for ids_per_label in all_ids]
    words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label] + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) for ids_per_label in all_ids]

    verbalizer.label_words_ids.data = torch.tensor(words_ids)
    verbalizer.words_ids_mask.data = torch.tensor(words_ids_mask)
    verbalizer.label_words_mask.data = torch.clamp(torch.tensor(words_ids_mask).sum(dim=-1), max=1)

    logger.info(f'After fixxing, the verbalizer should use \n{verbalizer.state_dict()}')

logger.info('Model loaded')

logger.info('Start testing')
if args.do_test:
    raw = trainer.predict(datasets['test'])
    result['best'] = postprocess_prediction(raw, processor.metrics)
logger.info('End testing')


if another_data:
    output_dir = os.path.join('outputs', DATA_PATH, args.experiment_name, ID)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'info.json'), 'w+') as f:
        json.dump({
            'id': ID,
            'args': args.__dict__,
            'results': {'test': result}
        }, f, indent=4)
    logger.info(f'Result saved to {output_dir}')

else:
    logger.info('For comparaison')
    logger.info(info['results']['test']['best']['metrics'])
    logger.info(result['best']['metrics'])

    assert abs(info['results']['test']['best']['metrics']['acc']-result['best']['metrics']['acc']) < EPSILON, "Inconsistent result"
    info['results']['test'].update(result)
    with open(os.path.join(DIR, 'info.json'), 'w+') as f:
        json.dump({
            'id': ID,
            'args': args.__dict__,
            'results': info['results']
        }, f, indent=4)
    logger.info(f'Result saved to {DIR}')
