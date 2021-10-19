import json
import argparse
import os
from copy import deepcopy
import logging
from typing import Tuple
from tqdm import tqdm
import timeit
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)
error_len = 0


class Data(object):
    def __init__(self, context, question, answer, answer_start):
        self.context = context
        self.question = question
        self.answer = answer
        self.answer_start = answer_start


class InputFeatures(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels


def read_data(args, path):

    with open(path, "rb") as f:
        data_dict = json.load(f)

    datas = []
    for ele in data_dict:
        if args.data_type == "SQuAD":
            answer_text = ""
            for answer in ele["answers"]:
                if answer_text == answer["text"]:
                    continue
                else:
                    answer_text = answer["text"]
                    datas.append(
                        Data(
                            context=ele["context"],
                            question=ele["question"],
                            answer=answer_text,
                            answer_start=answer["answer_start"],
                        )
                    )

        elif args.data_type == "RACE":
            datas.append(
                Data(
                    context=ele["context"],
                    question=ele["question"],
                    answer=ele["answer"],
                    answer_start=-1,
                )
            )

    return datas


def convert_data_to_features(args, tokenizer, datas, eval=False):
    global error_len
    features = []

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token

    num = 0
    for index, ele in enumerate(datas):
        try:
            context_tokens = tokenizer.tokenize(ele.context)
            answer_tokens = tokenizer.tokenize(ele.answer)
            question_tokens = tokenizer.tokenize(ele.question)
            answer_start = ele.answer_start

            max_context_length = args.max_seq_length - len(answer_tokens) - 3

            if len(context_tokens) > max_context_length:
                error_len += 1
                if answer_start == -1:
                    context_tokens = context_tokens[:max_context_length]
                else:
                    half_len = int(max_context_length / 2)
                    new_context = (
                        ele.context[half_len:answer_start]
                        + ele.context[answer_start:half_len]
                    )
                    context_tokens = tokenizer.tokenize(new_context)
                    if len(context_tokens) > max_context_length:
                        context_tokens = context_tokens[:max_context_length]

            num += 1

            input_tokens = [cls_token] + context_tokens + [sep_token]
            token_type_ids = [0] * len(input_tokens)
            input_tokens += answer_tokens + [sep_token]
            while len(token_type_ids) < len(input_tokens):
                token_type_ids.append(1)

            attention_mask = [1] * len(input_tokens)

            while len(input_tokens) < args.max_seq_length:
                input_tokens += [pad_token]
                token_type_ids.append(0)
                attention_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

            assert len(input_ids) == args.max_seq_length
            assert len(token_type_ids) == args.max_seq_length
            assert len(attention_mask) == args.max_seq_length

            question_tokens += [sep_token]
            while len(question_tokens) < args.max_query_length:
                question_tokens += [pad_token]

            label_ids = tokenizer.convert_tokens_to_ids(question_tokens)
            assert len(label_ids) == args.max_query_length

            if index < 20 and eval:
                logger.info("*** data features***")
                logger.info("tokens: %s" % " ".join(input_tokens))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in token_type_ids])
                )
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in attention_mask])
                )
                logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=label_ids,
                )
            )

        except Exception as e:
            print(e)
            raise e
            # continue

    return features


def evaluate(args, model, tokenizer, beam_size=1):

    start_time = timeit.default_timer()

    """ Load datas """
    eval_dataset = read_data(args, path=args.predict_file)

    cached_eval_features_file = "eval_{0}_{1}_{2}_{3}_{4}_{5}".format(
        args.model_name_or_path,
        str(args.data_type),
        str(args.max_seq_length),
        str(args.doc_stride),
        str(args.max_answer_length),
        str(args.max_query_length),
    )
    cached_eval_features_file = cached_eval_features_file.replace("/", "_")
    try:
        with open(cached_eval_features_file, "rb") as reader:
            eval_features = pickle.load(reader)
    except:
        eval_features = convert_data_to_features(
            args, tokenizer, eval_dataset, eval=True
        )
        logger.info(
            "  Saving eval features into cached file %s", cached_eval_features_file
        )
        with open(cached_eval_features_file, "wb") as writer:
            pickle.dump(eval_features, writer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in eval_features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in eval_features], dtype=torch.long
    )
    all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )

    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    # Evaluate!
    logger.info("***** Running Evaluate *****")
    logger.info("  Num orig examples = %d", len(eval_dataset))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  eval_batch_size = %d", args.eval_batch_size)

    num = 0
    res = []
    predict_questions_text = ""

    for step, batch in enumerate(tqdm(eval_dataloader)):

        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3],
        }

        if args.model_type in [
            "xlm",
            "roberta",
            "distilbert",
            "camembert",
            "bart",
            "longformer",
        ]:
            del inputs["token_type_ids"]

        if args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
            if args.version_2_with_negative:
                inputs.update({"is_impossible": batch[7]})
            if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                inputs.update(
                    {
                        "langs": (
                            torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id
                        ).to(args.device)
                    }
                )

        output_ids = model.generate(
            inputs["input_ids"],
            num_beams=beam_size,
            max_length=args.max_query_length,
            no_repeat_ngram_size=5,
            num_return_sequences=beam_size,
            # num_beam_groups=beam_size,
            early_stopping=True,
        )

        output_shape = output_ids.shape
        output_ids = output_ids.reshape(
            int(output_shape[0] / beam_size), beam_size, output_shape[1]
        )
        for predicts, groundtruth in zip(output_ids, inputs["labels"]):
            predict_questions = []
            groundtruth_text = tokenizer.decode(
                groundtruth,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for q_gen in predicts:
                prediction_text = tokenizer.decode(
                    q_gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                predict_questions.append(prediction_text)

            res.append(
                {
                    "predict_questions": predict_questions,
                    "ground_truth": groundtruth_text,
                }
            )

            if len(predict_questions) > 0:
                predict_questions_text += predict_questions[0] + "\n"
                num += 1
            else:
                predict_questions_text += "\n"

    evalTime = timeit.default_timer() - start_time

    logger.info(
        "Evaluation done %d in %f secs (%f sec per example)",
        num,
        evalTime,
        evalTime / num,
    )

    if "dev" in args.predict_file:
        data_type = "dev"
    elif "test" in args.predict_file:
        data_type = "test"
    elif "train" in args.predict_file:
        data_type = "train"
    else:
        data_type = "eval"

    output_file = os.path.join(
        args.output_dir, "{0}_beam_size_{1}".format(str(data_type), str(beam_size))
    )

    json.dump(res, open(output_file + ".json", "w"))
    with open(output_file + ".txt", "w") as file:
        file.write(predict_questions_text.strip())

    global error_len
    print(error_len)


def predict(args, model, tokenizer, features, beam_size=1):

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)
    token_type_ids = token_type_ids.to(args.device)

    output_ids = model.generate(
        input_ids,
        num_beams=beam_size,
        max_length=args.max_query_length,
        no_repeat_ngram_size=5,
        num_return_sequences=beam_size,
        # num_beam_groups=beam_size,
        early_stopping=True,
    )

    predict_questions = []
    for predict in output_ids:

        prediction_text = tokenizer.decode(
            predict, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predict_questions.append(prediction_text)

    return predict_questions


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
    # Other parameters
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--data_type",
        default="RACE",
        type=str,
        help="data type [SQuAD, RACE]",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluate.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
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
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=480,
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
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument("--beam_size", type=int, default=1, help="beam search size")

    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )

    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model.to(args.device)
    model.eval()

    if args.predict_file != None:
        evaluate(args, model=model, tokenizer=tokenizer, beam_size=args.beam_size)
    else:
        while 1:
            context = input("context: ")
            answer_text = input("answer: ")
            answer_start = context.find(answer_text)

            if answer_start != -1:

                features = convert_data_to_features(
                    args,
                    tokenizer,
                    [
                        Data(
                            context=context,
                            question="",
                            answer=answer_text,
                            answer_start=answer_start,
                        )
                    ],
                )
            else:
                features = convert_data_to_features(
                    args,
                    tokenizer,
                    [
                        Data(
                            context=context,
                            question="",
                            answer=answer_text,
                            answer_start=-1,
                        )
                    ],
                )
            result = predict(args, model, tokenizer, features, args.beam_size)
            print(result)


if __name__ == "__main__":
    main()
