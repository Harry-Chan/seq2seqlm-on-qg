# Two-Stage-Question-Generation
## Training

### Seq2Seq QG
#### SQuAD
```
python3 train_seq2seq_lm.py  --model_name_or_path facebook/bart-base  --data_type squad  --task_name  seq2seq_QG  --train_file data/squad_nqg/train_keywords.json  --dev_file data/squad_nqg/test_keywords.json  --predict_file data/squad_nqg/test_keywords.json  --batch_size 24  --epoch 5  --lr 5e-5  --output_dir squad_QG_seq2seq/
```
#### RACE
```
python3 train_seq2seq_lm.py  --model_name_or_path facebook/bart-base  --data_type race  --task_name  seq2seq_QG  --train_file data/race_eqg/train_keywords.json  --dev_file data/race_eqg/test_keywords.json  --predict_file data/race_eqg/test_keywords.json  --batch_size 24  --epoch 5  --lr 5e-5  --output_dir race_QG_seq2seq/
```


## Evaluation
Based on the package [`nlg-eval`](https://github.com/Maluuba/nlg-eval).
### install package
```
python3 setup_scorer.py
```