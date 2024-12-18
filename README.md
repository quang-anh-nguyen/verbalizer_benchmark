# Verbalizer benchmarking for text classification
This repository contains the code for benchmarking verbalizer baselines for text classification problems, published at LREC-COLING 2024, the Joint International Conference on Computational Linguistics, Language Resources and Evaluation. 20-25 MAY, 2024 / TORINO, ITALIA.

## :bookmark_tabs: Contents

**[💻 Requirments](##-Requirments)**

**[🚆 Training](##-Training)**

**[📋 Evaluation](##-Evaluation)**

**[🛠️ Customize](##-Customize)**

**[🔖 Citation](##-Citation)**




## 💻 Requirements
All requirements can be found in ```requirements.txt```. You can install required packages with ```pip install -r requirements.txt```

## 🚆 Training
To train a model in few-learning with verbalizers, simply run the following commands:

```shellscript
python3 main.py \\
$MODEL_TYPE \\
$MODEL_PATH \\
$DATASET \\
--experiment_name $EXPERIMENT_NAME\\
--seed $SEED \\
--do_train \\
--do_test \\
--do_zeroshot \\
--train_size $TRAIN_SIZE \\
--train_to_valid $TRAIN_TO_VALID \\
--split $SPLIT_0 $SPLIT_1 \\
--template_id $TEMPLATE_ID \\
--verbalizer_type $VERBALIZER_TYPE \\
--batchsize_train $BATCHSIZE_TRAIN \\
--batchsize_eval $BATCHSIZE_EVAL \\
--learning_rate 1e-5 $LEARNING_RATE \\
--epochs $EPOCHS \\
--device $DEVICE
```
where
* `$MODEL_TYPE` and `$MODEL_PATH` are the model name family (e.g. `roberta`) and the path of the pretrained model parsed to `transformers.ModelForMaskedLM.from_pretrained` (e.g. `roberta-large`).
* `$DATASET` is the dataset, see `data/processor.py` for more details.
* `$TRAIN_SIZE` is the number of train examples, `$TRAIN_TO_VALID` is the ratio train/validation examples. `$SPLIT0` is the random state for sampling labeled examples, `$SPLIT1` is the random state for spliting train and validation.
* `$TEMPLATE_ID` specifies the template to use, see `data/processor.py` for more details.
* `$VERBALIZER_TYPE` specifies the bverbalizer baseline, must be one of: `manual`, `soft`, `auto`.
* `$BATCHSIZE_TRAIN`, `$BATCHSIZE_EVAL`, `$LEARNING_RATE`, `$EPOCHS` are training parameters.
* `$DEVICE` is the device parsed to `torch.device`.

If `$VERBALIZER_TYPE` is `auto`, you can specify the number of label words per class via `--num_labelword`.

Termination of running the above command creates a folder with random-generated name `./outputs/$DATASET/$EXPERIMENT_NAME/XXXXXXXXXX` containing: 
* `checkpoint/`: folder resulted from `transformers.Trainer.save_model`.
* `info.json`: all information related to the run, including arguments, training history, predictions and metrics.
* `logfile.log`: log file.
* `verbalizer.json`: automatic label words if `$VERBALIZER_TYPE` is `auto`.

## 📋 Evaluation
To load and evaluate a pretrained model, run the following commands:

```
python3 evaluate.py \\
$DIR \\
--data_path $DATA_PATH
```
where
* `$DIR` is the directory created by training command above.
* `$DATA_PATH` is the test data path, if not identic to the test split of the training data.

By default, necessary arguments will be collected from `$DIR/info.json` for evaluation.

## 🛠️ Customize
To use your cusom template or verbalizer, create text files and parse them through `--template_file` or `--verbalizer_file` in the training command. The formatting of these files follow [template](https://thunlp.github.io/OpenPrompt/notes/template.html) and [verbalizer](https://thunlp.github.io/OpenPrompt/notes/verbalizer.html).

## 🔖 Citation

```
@inproceedings{nguyen2024enhancing,
  title={Enhancing Few-Shot Topic Classification with Verbalizers. A Study on Automatic Verbalizer and Ensemble Methods},
  author={Nguyen, Quang Anh and Tomeh, Nadi and Lebbah, Mustapha and Charnois, Thierry and Azzag, Hanane and Munoz, Santiago Cordoba},
  booktitle={LREC-COLING 2024. Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
  year={2024},
  month={5},
  day={20-25},
  address={Torino, Italia},
}
```
