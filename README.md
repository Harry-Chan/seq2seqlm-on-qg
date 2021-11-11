# Seq2seqLM-On-Question-Generation
## Training

### Seq2Seq QG
#### SQuAD
```
python3 train_seq2seq_QG.py \
  --model_type bart \
  --model_name_or_path facebook/bart-base \
  --do_train \
  --train_file data/squad_nqg/unilm_train_keywords.json \
  --data_type SQuAD \
  --per_gpu_train_batch_size 25 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --logging_steps 100 \
  --save_steps -1 \
  --output_dir bart_QG_SQuAD/
```
#### RACE
```
python3 train_seq2seq_QG.py \
  --model_type bart \
  --model_name_or_path facebook/bart-base \
  --do_train \
  --train_file data/race_eqg/race_train_keywords_17445.json \
  --data_type RACE \
  --per_gpu_train_batch_size 25 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --logging_steps 100 \
  --save_steps -1 \
  --output_dir bart_QG_RACE/
```

## Generating

### Seq2Seq QG
#### SQuAD
```
python3 predict_seq2seq_QG.py \
  --model_type bart \
  --model_name_or_path bart_QG_SQuAD/epoch-19/ \
  --eval_batch_size 8 \
  --beam_size 3 \
  --output_dir bart_QG_SQuAD/epoch-19/ \
  --predict_file data/squad_nqg/unilm_test_keywords.json \
  --data_type SQuAD
```
#### RACE
```
python3 predict_seq2seq_QG.py \
  --model_type bart \
  --model_name_or_path bart_QG_RACE/epoch-19/ \
  --eval_batch_size 8 \
  --beam_size 3 \
  --output_dir bart_QG_RACE/epoch-19/ \
  --predict_file data/race_eqg/race_test_keywords.json \
  --data_type RACE
```

## Evaluation
Based on the package [`nlg-eval`](https://github.com/Maluuba/nlg-eval).
### install package
```
python3 setup_scorer.py
```
### Evaluate on SQuAD
```
nlg-eval --hypothesis=bart_QG_SQuAD/epoch-19/test_beam_size_3.txt --references=data/squad_nqg/nqg_tgt-test.txt --no-skipthoughts  --no-glove
```

### Evaluate on RACE
```
nlg-eval --hypothesis=bart_QG_RACE/epoch-19/test_beam_size_3.txt --references=data/race_eqg/race_test_q.txt --no-skipthoughts  --no-glove
```