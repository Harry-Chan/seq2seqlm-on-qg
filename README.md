# Seq2seqLM-On-Question-Generation
Use Seq2seqLM (BART) on Question  Generation Task

## Training Strategy & Datasets
We report three dataset on two training strategy

### Highlight Training Strategy and Extract-based QA Dataset (The answer is a span in context)
The model input sequence `X` of the "Highlight Training Strategy" is as follows
```
X = [c1, c2, ..., [HL], a1, ..., a|A|, [HL], ..., c|C|]
```
> Proposed by [Ying-Hong Chan & Yao-Chung Fan. (2019). A Re-current BERT-based Model for Question Generation.](https://www.aclweb.org/anthology/D19-5821/)

#### SQuAD NQG
- train: 75722
- test: 11877
> [Learning to Ask: Neural Question Generation for Reading Comprehension](https://arxiv.org/abs/1705.00106)

#### DRCD
- train: 26936
- test: 3493
> [DRCD: a Chinese Machine Reading Comprehension Dataset](https://arxiv.org/abs/1806.00920)

### Naive Training Strategy and Abstract-based QA Dataset
The model input sequence `X` of the "Naive Training Strategy" is as follows
```
X = [c1, c2, ..., c|C|, [SEP], a1, ..., a|A|]
```

#### RACE EQG
- train: 17445
- test: 950
> [EQG-RACE: Examination-Type Question Generation](https://arxiv.org/abs/2012.06106)



## Environment Setup
1. Install packages `pip install -r requirements.txt`

2. Setup scorer `python setup_scorer.py`

## Training
#### SQuAD
```
python train_seq2seq_lm.py \
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

#### DRCD
```
python train_seq2seq_lm.py \
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


#### RACE
```
python train_seq2seq_lm.py \
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

## Generating
#### SQuAD
```
python train_seq2seq_lm.py \
  --model_name_or_path facebook/bart-base \
  --data_type squad \
  --task_name seq2seq_QG \
  --predict_file data/squad_nqg/test.json \
  --output_dir squad_QG_seq2seq \
  --run_test \
  --from_checkpoint squad_QG_seq2seq/checkpoint/
```


#### DRCD
```
python train_seq2seq_lm.py \
  --model_name_or_path uer/bart-base-chinese-cluecorpussmall \
  --data_type drcd \
  --task_name  seq2seq_QG \
  --predict_file data/drcd/test.json \
  --output_dir drcd_QG_seq2seq \
  --run_test \
  --from_checkpoint drcd_QG_seq2seq/checkpoint/
```


#### RACE
```
python train_seq2seq_lm.py \
  --model_name_or_path facebook/bart-base \
  --data_type race \
  --task_name  seq2seq_QG \
  --predict_file data/race_eqg/test.json \
  --output_dir race_QG_seq2seq \
  --run_test \
  --from_checkpoint race_QG_seq2seq/checkpoint/
```


## Expriments
We report score with [NQG Scorer](https://github.com/xinyadu/nqg) which is using in SQuAD NQG

Default BART-HLQG and BART-QG sizes are "base"


### SQuAD NQG
Model                                                                  |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
-----------------------------------------------------------------------|------|------|------|------|------|-------|
BERT-HLSQG [(Chan, et al.)](https://www.aclweb.org/anthology/D19-5821/) |49.73 |34.60 |26.13 |20.33 |23.88 |48.23  |
BART-HLQG                                                             |56.57 |40.25 |30.62 |23.88 |25.64 |51.68  |

```
python nqg/qgevalcap/eval.py \
  --src data/squad_nqg/src-test.txt \
  --tgt data/squad_nqg/tgt-test.txt \
  --out data/outputs/BART_HLQG_predict_squad.txt
```

### DRCD
Model                                                                  |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
-----------------------------------------------------------------------|------|------|------|------|------|-------|
BART-HLQG                                                              |55.26 |45.85 |39.35 |34.36 |28.45 |50.38  |

```
python nqg/qgevalcap/eval.py \
  --src data/drcd/src-test.txt \
  --tgt data/drcd/tgt-test.txt \
  --out data/outputs/BART_HLQG_predict_drcd.txt
```
### RACE EQG
Model                                                                  |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
-----------------------------------------------------------------------|------|------|------|------|------|-------|
Unified model + ELMo [(Xin, et al)](https://arxiv.org/abs/2012.06106)  |35.10 |21.08 |15.19 |11.96 |14.94 |34.24  |
BART-QG                                                                |46.73 |32.74 |25.20 |20.18 |22.58 |47.23  |

```
python nqg/qgevalcap/eval.py \
  --src data/race_eqg/src-test.txt \
  --tgt data/race_eqg/tgt-test.txt \
  --out data/outputs/BART_QG_predict_race.txt
```
