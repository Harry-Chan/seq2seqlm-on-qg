import json
import argparse
import os
import random
from copy import deepcopy
import logging
import pickle
from tqdm import tqdm, trange
import timeit

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import (AdamW, get_linear_schedule_with_warmup,
                        AutoModelForMaskedLM, AutoConfig, AutoTokenizer)
from transformers.trainer_utils import is_main_process

import wandb

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class Data(object):
    def __init__(self,
                 context,
                 question,
                 answer,
                 keywords):
        self.context = context
        self.question = question
        self.answer = answer
        self.keywords = keywords

class InputFeatures(object):
    def __init__(self, 
                 input_ids, 
                 token_type_ids, 
                 attention_mask, 
                 labels, 
                 label_indexs):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.label_indexs = label_indexs

def read_data(args, path):

    with open(path, 'rb') as f:
        data_dict = json.load(f)

    datas = []
    for ele in data_dict:
        if args.data_type == 'SQuAD':
            answer_text = ''
            for answer in ele['answers']:
                if answer_text == answer['text']:
                    continue
                else:
                    answer_text = answer['text']
                    datas.append(
                        Data(context = ele['context'],
                             question = ele['question'],
                             answer = answer_text,
                             keywords = ele['noun_keywords'])
                        )

        elif args.data_type == 'RACE':
            datas.append(
                Data(context = ele['context'],
                     question = ele['question'],
                     answer = ele['answer'],
                     keywords = ele['noun_keywords'])
                )

    return datas


def convert_data_to_features(args, datas, tokenizer):

    features = []
    
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token
    pad_token = tokenizer.pad_token

    num = 0

    for index, ele in enumerate(tqdm(datas)):
        try:
            context_tokens = tokenizer.tokenize(ele.context)
            answer_tokens = tokenizer.tokenize(ele.answer)
            question_tokens = tokenizer.tokenize(ele.question)

            if len(answer_tokens) > args.max_answer_length:
                continue
                # answer_tokens = answer_tokens[0:args.max_answer_length]  

            if len(question_tokens) > args.max_query_length:
                continue
                # question_tokens = question_tokens[0:args.max_query_length] 
            
            keyword_indexs = []
            keyword_tokens = []
            keyword_tokens_num = 0
            if len(ele.keywords) > 0:
                keyword_num = random.randint(1, len(ele.keywords)) 
                if keyword_num != 0:
                    while (1):
                        if len(keyword_indexs) == keyword_num:
                            break          
                        random_index = random.randint(0, len(ele.keywords) - 1)
                        if random_index not in keyword_indexs:
                            keyword_indexs.append(random_index)

                    keyword_indexs = sorted(keyword_indexs, key=lambda x: x)

                    for i in keyword_indexs:
                        token = tokenizer.tokenize(' ' + ele.keywords[i])
                        keyword_tokens.append(token)
                        keyword_tokens_num += len(token) + 1
                else:
                    keyword_tokens_num = 2  #<pad> </s>
            else:
                keyword_tokens_num = 2  #<pad> </s>
            
            
            max_context_length = args.max_seq_length - len(answer_tokens) - (keyword_tokens_num) - len(question_tokens) - 4
            if len(context_tokens) > max_context_length:
                continue
                # context_tokens = context_tokens[0:max_context_length]
            num+=1

            input_tokens = [cls_token] + context_tokens + [sep_token]
            token_type_ids = [0] * len(input_tokens)

            input_tokens += answer_tokens + [sep_token]
            while len(token_type_ids) < len(input_tokens):
                token_type_ids.append(1)

            if len(keyword_tokens) == 0:
                input_tokens += [pad_token] + [sep_token]
            else:
                for token in keyword_tokens:
                    input_tokens += token + [sep_token]
            while len(token_type_ids) < len(input_tokens):
                token_type_ids.append(0)  

            attention_mask = [1] * len(input_tokens)

            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

            question_tokens += [sep_token]
            question_ids = tokenizer.convert_tokens_to_ids(question_tokens)

            for question_id in question_ids:               
                SQ_input_ids = deepcopy(input_ids)
                SQ_token_type_ids = deepcopy(token_type_ids)
                SQ_attention_mask = deepcopy(attention_mask)

                SQ_input_ids.append(tokenizer.convert_tokens_to_ids(mask_token))
                SQ_token_type_ids.append(1)
                SQ_attention_mask.append(1)

                label_indexs = len(SQ_input_ids) - 1
                label_ids = [-100] * args.max_seq_length
                label_ids[label_indexs] = question_id

                while len(SQ_input_ids) < args.max_seq_length:
                  SQ_input_ids.append(0)
                  SQ_token_type_ids.append(0)
                  SQ_attention_mask.append(0)

                assert len(SQ_input_ids) == args.max_seq_length
                assert len(SQ_token_type_ids) == args.max_seq_length
                assert len(SQ_attention_mask) == args.max_seq_length
                assert len(label_ids) == args.max_seq_length 

                if index < 20 :
                    logger.info("*** data features***")
                    logger.info("tokens: %s" % " ".join(input_tokens))
                    logger.info("keywords: %s" % " ".join([str(x) for x in keyword_tokens]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in SQ_input_ids]))
                    logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in SQ_token_type_ids]))                    
                    logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in SQ_attention_mask]))
                    logger.info("output_ids: %s" % " ".join([str(x) for x in question_ids]))
                    logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                    logger.info("label_indexs: %d" ,label_indexs)
                    # input()
                features.append(
                    InputFeatures(
                        input_ids = SQ_input_ids,
                        attention_mask = SQ_attention_mask,
                        token_type_ids = SQ_token_type_ids,
                        labels = label_ids,
                        label_indexs = label_indexs
                        ))
                input_ids.append(question_id)
                token_type_ids.append(1)
                attention_mask.append(1)

        except Exception as e:
            print(e)
            raise e
            # continue
    print('data_num:',num)
    input()
    return features


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, model, tokenizer):

    """ Load datas """
    train_dataset = read_data(args, args.train_file)

    cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}_{4}'.format(
    args.model_name_or_path, str(args.max_seq_length), str(args.doc_stride), str(args.max_answer_length), str(args.max_query_length))
    cached_train_features_file = cached_train_features_file.replace('/','_')
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    except:
        train_features = convert_data_to_features(args, train_dataset, tokenizer)
        logger.info("  Saving train features into cached file %s", cached_train_features_file)
        with open(cached_train_features_file, "wb") as writer:
            pickle.dump(train_features, writer)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)
                    
    train_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_features)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_dataset))
    logger.info("  Num split examples = %d", len(train_features))    
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    
    # Init wandb 
    wandb.init(project="S_KPQG",
        name="{0}_{1}_{2}_{3}_{4}".format(str(args.model_name_or_path),str(args.max_seq_length), str(args.doc_stride), str(args.max_answer_length), str(args.max_query_length))
        )
    wandb.watch(model)

    for epoch, _ in enumerate(train_iterator):
        print("epoch",epoch)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)                    
                    wandb.log({"lr": scheduler.get_lr()[0], "step": global_step})
                    train_loss = (tr_loss - logging_loss) / args.logging_steps
                    wandb.log({"loss": train_loss, "step": global_step})
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        eval_loss = evaluate(args, model, tokenizer)
                        wandb.log({"eval_loss": eval_loss, "step": global_step})

                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        wandb.log({"lr": scheduler.get_lr()[0]})
        train_loss = tr_loss / global_step
        wandb.log({"epoch_loss": train_loss, "epoch": epoch})
        if args.local_rank == -1 and args.evaluate_during_training:
            eval_loss = evaluate(args, model, tokenizer)
            wandb.log({"epoch_eval_loss": eval_loss, "epoch": epoch})

        output_dir = os.path.join(args.output_dir, "epoch-{}".format(epoch))
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model epoch-%d to %s", epoch, output_dir)        

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):

    """ Load datas """
    eval_dataset = read_data(args.predict_file)

    cached_eval_features_file = args.predict_file+'_{0}_{1}_{2}_{3}_{4}'.format(
    args.model_name_or_path, str(args.max_seq_length), str(args.doc_stride), str(args.max_answer_length), str(args.max_query_length))
    try:
        with open(cached_eval_features_file, "rb") as reader:
            eval_features = pickle.load(reader)
    except:
        eval_features = convert_data_to_features(args, eval_dataset, tokenizer)
        logger.info("  Saving eval features into cached file %s", cached_eval_features_file)
        with open(cached_eval_features_file, "wb") as writer:
            pickle.dump(eval_features, writer)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
    all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
                    
    eval_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num orig examples = %d", len(eval_dataset))
    logger.info("  Num split examples = %d", len(eval_features)) 
    logger.info("  Batch size = %d", args.eval_batch_size)

    tr_eval_loss = 0
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],                
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)

            eval_loss = outputs[0]

            if args.n_gpu > 1:
                eval_loss = eval_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            tr_eval_loss += eval_loss.item()

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(eval_dataloader))
    logger.info("  Evaluation loss %f", tr_eval_loss / len(eval_dataloader))

    return tr_eval_loss / len(eval_dataloader)



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type bert",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--data_type",
        default=None,
        type=str,
        required=True,
        help="SQuAD or RACE",
    )
    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    global_step, tr_loss = train(args, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()