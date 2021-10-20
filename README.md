# Two-Stage-Question-Generation
## Training

### Seq2Seq QG
#### SQuAD
```
python3 train_seq2seq_QG.py \
  --model_type bart \
  --model_name_or_path facebook/bart-base \
  --do_train \
  --train_file data/squad_v1.1/unilm_train_keywords.json \
  --data_type SQuAD \
  --per_gpu_train_batch_size 25 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --logging_steps 100 \
  --save_steps -1 \
  --output_dir bart_QG_SQuAD/ \
  --warmup_steps 100 
```
#### RACE
```
python3 train_seq2seq_QG.py \
  --model_type bart \
  --model_name_or_path facebook/bart-base \
  --do_train \
  --train_file data/race/race_train_keywords_17445.json \
  --data_type RACE \
  --per_gpu_train_batch_size 25 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --logging_steps 100 \
  --save_steps -1 \
  --output_dir bart_QG_RACE/ \
  --warmup_steps 100 
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
  --predict_file data/squad_v1.1/unilm_test_keywords.json \
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
  --predict_file data/race/race_test_keywords.json \
  --data_type RACE
```

## Evaluation
Based on the package [`nlg-eval`](https://github.com/Maluuba/nlg-eval).
### install package
```
nlg-eval --setup
```
### install java jdk
```
sudo apt-get update
sudo apt-get install default-jdk -y
```
### using on SQuAD
```
nlg-eval --hypothesis=bart_QG_SQuAD/epoch-19/test_beam_size_3.txt --references=data/suqad_v1.1/nqg_tgt-test.txt
```

### using on RACE
```
nlg-eval --hypothesis=bart_QG_RACE/epoch-19/test_beam_size_3.txt --references=data/race/race_text_q.txt
```