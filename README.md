# Seq2seqLM-On-Question-Generation
## Training

### Seq2Seq QG
#### SQuAD
```
python3 train_seq2seq_lm.py \
  --model_name_or_path facebook/bart-base \
  --data_type squad \
  --task_name  seq2seq_QG \
  --train_file data/squad_nqg/train.json \
  --dev_file data/squad_nqg/test.json \
  --predict_file data/squad_nqg/test.json \
  --batch_size 24 \
  --epoch 10 \
  --lr 5e-5 \
  --output_dir squad_QG_seq2seq/ \
  --wandb_logging_steps  100
```
#### RACE
```
python3 train_seq2seq_lm.py \
  --model_name_or_path facebook/bart-base \
  --data_type race \
  --task_name  seq2seq_QG \
  --train_file data/race_eqg/train.json \
  --dev_file data/race_eqg/test.json \
  --predict_file data/race_eqg/test.json \
  --batch_size 24 \
  --epoch 20 \
  --lr 5e-5 \
  --output_dir race_QG_seq2seq/ \
  --wandb_logging_steps  100
```
#### DRCD
```
python3 train_seq2seq_lm.py \
  --model_name_or_path uer/bart-base-chinese-cluecorpussmall \
  --data_type drcd \
  --task_name  seq2seq_QG \
  --train_file data/drcd/train.json \
  --dev_file data/drcd/test.json \
  --predict_file data/drcd/test.json \
  --batch_size 24 \
  --epoch 20 \
  --lr 5e-5 \
  --output_dir drcd_QG_seq2seq/ \
  --wandb_logging_steps  100
```
## Generating
#### SQuAD
```
python3 train_seq2seq_lm.py \
  --model_name_or_path facebook/bart-base \
  --data_type squad \
  --task_name seq2seq_QG \
  --predict_file data/squad_nqg/test.json \
  --output_dir squad_QG_seq2seq \
  --run_test \
  --from_checkpoint squad_QG_seq2seq/checkpoint/
```

#### RACE
```
python3 train_seq2seq_lm.py \
  --model_name_or_path facebook/bart-base \
  --data_type race \
  --task_name  seq2seq_QG \
  --predict_file data/race_eqg/test.json \
  --output_dir race_QG_seq2seq \
  --run_test \
  --from_checkpoint race_QG_seq2seq/checkpoint/
```

#### DRCD
```
python3 train_seq2seq_lm.py \
  --model_name_or_path uer/bart-base-chinese-cluecorpussmall \
  --data_type drcd \
  --task_name  seq2seq_QG \
  --predict_file data/drcd/test.json \
  --output_dir drcd_QG_seq2seq \
  --run_test \
  --from_checkpoint drcd_QG_seq2seq/checkpoint/
```



## Evaluation
Based on the package [`nlg-eval`](https://github.com/Maluuba/nlg-eval).
### install package
```
python3 setup_scorer.py
```
