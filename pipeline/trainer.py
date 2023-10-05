import os
import torch
import logging
import inspect

from torch.nn import Linear
from dataclasses import dataclass
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.nn import CrossEntropyLoss

from pipeline.loader import get_prompt_dataloader, transpose
from data.data import get_metric
from utils import get_logger

LOSS_FUNCTION = {
    'ce': CrossEntropyLoss
}

DIVIDE_BY = 100

# logger = logging.getLogger(__name__)
logger = get_logger(__name__)
# logger = get_logger('transformers.trainer.'+__name__)

def get_training_arguments(args):
    kwargs = {
        "output_dir": os.path.join(args.output_dir, args.dataset, args.experiment_name, args.experiment_id),
        "do_train": args.do_train,
        "evaluation_strategy": "steps",
        "eval_steps": args.valid_every_steps,
        "per_device_train_batch_size": args.batchsize_train,
        "per_device_eval_batch_size": args.batchsize_eval,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_accumulation_steps": 1,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
        "num_train_epochs": args.epochs,
        "warmup_ratio": 0.1,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": args.valid_every_steps,
        "save_total_limit": 1,
        "seed": args.seed,
        "run_name": (args.experiment_name+'.'+args.experiment_id) if args.experiment_name is not None else args.experiment_id,
        # "label_names": ['label'],
        "load_best_model_at_end": True,
        "metric_for_best_model": 'eval_score',
        "optim": 'adamw_torch'
    }

    if args.verbalizer_type=='manual':
        kwargs["second_learning_rate"]=args.second_learning_rate
    
    return CustomTrainingArguments(**kwargs)

def get_evaluating_arguments(args):
    kwargs = {
        "output_dir": os.path.join(args.output_dir, args.dataset, args.experiment_name, args.experiment_id),
        "per_device_train_batch_size": args.batchsize_train,
        "per_device_eval_batch_size": args.batchsize_eval,
        "eval_accumulation_steps": 1,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_total_limit": 1,
        "seed": args.seed,
        "run_name": (args.experiment_name+'.'+args.experiment_id) if args.experiment_name is not None else args.experiment_id,
        # "label_names": ['label'],
    }
    
    return CustomTrainingArguments(**kwargs)

@dataclass
class CustomTrainingArguments(TrainingArguments):

    template_type: str='manual'
    verbalizer_type: str='manual'

    second_learning_rate: int=1e-3


class BaseTrainer(Trainer):
    def __init__(
            self, 
            model=None, 
            args=None,
            datasets=None,
            prompter=None,
            loss_function='ce',
            **kwargs
        ):

        self.text_datasets = datasets
        self.processor = prompter['processor']
        self.template = prompter['template']
        self.verbalizer = prompter['verbalizer']
        self.plm_wrapper = prompter['plm_wrapper']
        self.tokenizer = prompter['tokenizer']

        def compute_metrics(evalpredictions):
            predictions = evalpredictions.predictions
            labels = evalpredictions.label_ids
            loss = self.loss_function(torch.from_numpy(predictions), torch.from_numpy(labels))
            if len(predictions.shape)>=2:
                predictions = predictions.argmax(-1)
            result = {'score': 0.}
            for metric in self.processor.metrics:
                result[metric] = get_metric(metric)(labels, predictions)
                result['score'] += result[metric]
            result['score'] = result['score']/len(self.processor.metrics) - loss/DIVIDE_BY
            return result
        
        def preprocess_logits(logits, labels):
            # Keep logits to do ensemble and view distribution
            return logits
            return logits.argmax(-1)


        super().__init__(
            model=model,
            args=args,
            train_dataset=self.wrap_tokenize(datasets['train']) if 'train' in datasets  else None,
            eval_dataset=self.wrap_tokenize(datasets['valid']) if 'valid' in datasets else None,
            tokenizer=prompter['tokenizer'],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits,
            **kwargs
        )
        self.loss_function = LOSS_FUNCTION[loss_function]()
        
    def wrap_tokenize(self, dataset):
        prompt_loader = get_prompt_dataloader(self.processor, dataset, self.template, self.tokenizer, self.plm_wrapper)
        tokenized = prompt_loader.tensor_dataset
        return tokenized

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            return super().get_eval_dataloader()
        return super().get_eval_dataloader(self.wrap_tokenize(eval_dataset))
    
    def get_test_dataloader(self, test_dataset):
        return super().get_test_dataloader(self.wrap_tokenize(test_dataset))
     
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels']
        outputs = model(**inputs)
        logits = outputs['logits']
        loss = self.loss_function(logits, labels)
        if return_outputs:
            return loss, outputs
        return loss     
        
class SoftVerbalizerTrainer(BaseTrainer):

    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_groups = [
            {
                "params": [p for n, p in self.model.prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": [p for n, p in self.model.prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.,
                "lr": self.args.learning_rate
            },
            {
                "params": [] if isinstance(self.model.verbalizer.head, Linear) else [p for n, p in self.model.verbalizer.head.named_parameters() if (self.model.verbalizer.head_last_layer_full_name not in n) and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": [] if isinstance(self.model.verbalizer.head, Linear) else [p for n, p in self.model.verbalizer.head.named_parameters() if (self.model.verbalizer.head_last_layer_full_name not in n) and any(nd in n for nd in no_decay)],
                "weight_decay": 0.,
                "lr": self.args.learning_rate
            },
            {
                "params": [self.model.head.neamed_parameters()] if isinstance(self.model.verbalizer.head, Linear) else [p for n, p in self.model.verbalizer.head.named_parameters() if (self.model.verbalizer.head_last_layer_full_name in n) and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.second_learning_rate
            },
            {
                "params": [self.model.head.neamed_parameters()] if isinstance(self.model.verbalizer.head, Linear) else [p for n, p in self.model.verbalizer.head.named_parameters() if (self.model.verbalizer.head_last_layer_full_name in n) and any(nd in n for nd in no_decay)],
                "weight_decay": 0.,
                "lr": self.args.second_learning_rate
            }
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(params_groups, **optimizer_kwargs)

        return self.optimizer

class AutomaticVerbalizerCallback(TrainerCallback):

    # def __init__(self, model, **kwargs):
    #     super().__init__()
    #     self.model = model

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Callback for optimize_to_initialize with search_id={kwargs['model'].verbalizer.search_id}", __name__)
        logger.info(f"Callback for optimize_to_initialize with search_id={kwargs['model'].verbalizer.search_id}")
        kwargs['model'].verbalizer.optimize_to_initialize()


class AutomaticVerbalizerTrainer(BaseTrainer):
    
    def __init__(self, model=None, args=None, datasets=None, prompter=None, **kwargs):
        super().__init__(model, args, datasets, prompter, callbacks=[AutomaticVerbalizerCallback()], **kwargs)

class KnowledgeableVerbalizerTrainer(BaseTrainer):
    pass
    def __init__(args, **kwargs):
        super().__init__(args, **kwargs)
        



def get_trainer_cls_kwargs(args):
    if args.verbalizer_type=='soft':
        trainer_cls = SoftVerbalizerTrainer
    elif args.verbalizer_type=='auto':
        trainer_cls = AutomaticVerbalizerTrainer
    else:
        trainer_cls = BaseTrainer

    trainer_kwargs = list(inspect.signature(trainer_cls.__init__).parameters.keys())

    return trainer_cls, trainer_kwargs

def get_trainer(parser_args, **kwargs):
    trainer_cls, trainer_kwargs = get_trainer_cls_kwargs(parser_args)

    return trainer_cls(**{kw: kwargs[kw] for kw in kwargs if kw in trainer_kwargs})
