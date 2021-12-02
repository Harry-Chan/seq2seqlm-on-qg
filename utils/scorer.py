from nlgeval import NLGEval
from collections import defaultdict
import os
import stanza
import json
import re
import string


class ModelEvalMixin:
    def __init__(self, preprocess=True):
        self.preprocess = preprocess
        self.nlgeval = NLGEval(
            no_glove=True, no_skipthoughts=True, metrics_to_omit=["CIDEr"]
        )
        self.score = defaultdict(lambda: 0.0)
        self.len = 0
        if self.preprocess:
            self.nlp = stanza.Pipeline(
                lang="en", processors="tokenize", tokenize_no_ssplit=True, verbose=False
            )

    def write_predict(self, decode_question, ref_question, data_type):
        #
        log_dir = (
            os.path.join(self.trainer.default_root_dir, "dev")
            if self.trainer.log_dir is None
            else self.trainer.log_dir
        )
        os.makedirs(log_dir, exist_ok=True)

        # write results
        with open(
            os.path.join(log_dir, "predict.jsonl"), "a", encoding="utf-8"
        ) as log_f:
            log_f.write(
                json.dumps(
                    {"hyp": decode_question, "ref": ref_question}, ensure_ascii=False
                )
                + "\n"
            )
        # write log for dataset type
        if data_type == "squad":
            # rewrite for nqg_squad type
            with open(
                os.path.join(log_dir, "predict_for_scorer.txt"),
                "a",
                encoding="utf-8",
            ) as log_f:
                decode_question = decode_question.lower()
                decode_question = decode_question.replace("?", " ?")
                decode_question = decode_question.replace(",", " ,")
                decode_question = decode_question.replace("'s", " 's")
                decode_question = decode_question.replace("...", " ...")

                # replace string: "hello" world -> `` hello '' world
                decode_question = re.sub(' "', "``", decode_question)
                decode_question = re.sub('"', "''", decode_question)
                decode_question = decode_question.replace("``", " `` ")
                decode_question = decode_question.replace("''", " ''")
                log_f.write(decode_question + "\n")
        elif data_type == "race":
            # rewrite for eqg_race type
            with open(
                os.path.join(log_dir, "predict_for_scorer.txt"),
                "a",
                encoding="utf-8",
            ) as log_f:
                decode_question = decode_question.lower()
                decode_question = decode_question.replace("?", " ?")
                decode_question = decode_question.replace(",", " ,")
                decode_question = decode_question.replace("'s", " 's")
                decode_question = decode_question.replace("n't", " n't")
                log_f.write(decode_question + "\n")
        elif data_type == "drcd":
            # rewrite for drcd type
            with open(
                os.path.join(log_dir, "predict_for_scorer.txt"),
                "a",
                encoding="utf-8",
            ) as log_f:

                # replace string: 台灣哪一行政區的月均最高與最低氣溫相差21.2度？ -> 台 灣 哪 一 行 政 區 的 月 均 最 高 與 最 低 氣 溫 相 差 21 . 2 度 ？

                decode_question = (
                    decode_question.replace(" ", "").replace("?", "？").replace("•", "・")
                )

                punc = string.punctuation

                rewrite_question = ""
                for i in range(len(decode_question)):

                    if (
                        "\u4e00" <= decode_question[i] <= "\u9fa5"
                        or decode_question[i] == "？"
                        or decode_question[i] == "、"
                        or decode_question[i] in punc
                    ):
                        rewrite_question += decode_question[i] + " "
                    else:
                        rewrite_question += decode_question[i]
                        if i + 1 != len(decode_question) and (
                            "\u4e00" <= decode_question[i + 1] <= "\u9fa5"
                            or decode_question[i + 1] == "？"
                            or decode_question[i + 1] == "、"
                            or decode_question[i + 1] in punc
                        ):
                            rewrite_question += " "
                print(rewrite_question.strip())
                log_f.write(rewrite_question.strip() + "\n")

    def evaluate_predict(self, data_type):

        log_dir = (
            os.path.join(self.trainer.default_root_dir, "dev")
            if self.trainer.log_dir is None
            else self.trainer.log_dir
        )

        # load predict and add score
        # with open(
        #     os.path.join(log_dir, "predict.jsonl"), "r", encoding="utf-8"
        # ) as log_f:
        #     lines = log_f.readlines()
        #     for i, line in enumerate(lines):
        #         line = json.loads(line)
        #         hyp = line["hyp"]
        #         ref = line["ref"]
        #         scorer.add(hyp, [ref])
        #     scorer.compute(save_score_report_path=log_dir)

        # nlg-eval scorer
        # nlg_predict_file_path = os.path.join(log_dir, "predict_for_scorer.txt")
        # nlg_predict_score_out_path = os.path.join(log_dir, "nlg-eval_scorer.txt")
        # if data_type == "squad":
        #     os.system(
        #         "nlg-eval --references=data/squad_nqg/nqg_tgt-test.txt  --hypothesis=%s  --no-skipthoughts  --no-glove  >> %s"
        #         % (nlg_predict_file_path, nlg_predict_score_out_path)
        #     )
        # elif data_type == "race":
        #     os.system(
        #         "nlg-eval --references=data/race_eqg/race_test_q.txt  --hypothesis=%s  --no-skipthoughts  --no-glove >> %s"
        #         % (nlg_predict_file_path, nlg_predict_score_out_path)
        #     )
        # elif data_type == "drcd":
        #     os.system(
        #         "nlg-eval --references=data/drcd/drcd_test_q.txt  --hypothesis=%s  --no-skipthoughts  --no-glove >> %s"
        #         % (nlg_predict_file_path, nlg_predict_score_out_path)
        #     )
        
        # nqg scorer
        assert os.path.isdir(
            "nqg"
        ), 'nqg scorer is not detect, please check "README.md" for help'
        nqg_predict_file_path = os.path.join(log_dir, "predict_for_scorer.txt")
        nqg_predict_score_out_path = os.path.join(log_dir, "nqg_scorer.txt")
        if data_type == "squad":
            os.system(
                "python nqg/qgevalcap/eval.py --src data/squad_nqg/src-test.txt --tgt data/squad_nqg/tgt-test.txt --out %s >> %s"
                % (nqg_predict_file_path, nqg_predict_score_out_path)
            )
        elif data_type == "race":
            os.system(
                "python nqg/qgevalcap/eval.py --src data/race_eqg/src-test.txt --tgt data/race_eqg/tgt-test.txt --out %s >> %s"
                % (nqg_predict_file_path, nqg_predict_score_out_path)
            )
        elif data_type == "drcd":
            os.system(
                "python nqg/qgevalcap/eval.py --src data/drcd/src-test.txt --tgt data/drcd/tgt-test.txt --out %s >> %s"
                % (nqg_predict_file_path, nqg_predict_score_out_path)
            )
        print("see log_dir:`%s` for full report" % log_dir)

    def save_huggingface_model(self):
        log_dir = (
            os.path.join(self.trainer.default_root_dir, "dev")
            if self.trainer.log_dir is None
            else self.trainer.log_dir
        )
        log_dir = os.path.join(log_dir, "huggingface_model")
        os.makedirs(log_dir, exist_ok=True)
        self.model.save_pretrained(log_dir)
        self.tokenizer.save_pretrained(log_dir)

    # def _preprocess(self, raw_sentence):
    #     result = self.nlp(raw_sentence.replace("\n\n", ""))
    #     tokens = []
    #     try:
    #         for token in result.sentences[0].tokens:
    #             tokens.append(token.text.lower())
    #             tokenize_sentence = " ".join(tokens)
    #     except:
    #         print('_preprocess fail, return ""\n', raw_sentence, result)
    #         return ""
    #     return tokenize_sentence

    # def add(self, hyp, refs):
    #     refs = refs[:]
    #     if self.preprocess:
    #         hyp = self._preprocess(hyp)
    #         refs = [self._preprocess(ref) for ref in refs]
    #         print(hyp, refs)
    #     score = self.nlgeval.compute_individual_metrics(hyp=hyp, ref=refs)
    #     for score_key in score.keys():
    #         self.score[score_key] += score[score_key]
    #     self.len += 1

    # def compute(self, save_score_report_path=None):
    #     if save_score_report_path is not None:
    #         os.makedirs(save_score_report_path, exist_ok=True)
    #         score_f = open(
    #             os.path.join(save_score_report_path, "our_scorer.txt"),
    #             "w",
    #             encoding="utf-8",
    #         )
    #     for score_key in self.score.keys():
    #         _score = self.score[score_key] / self.len
    #         if save_score_report_path is not None:
    #             score_f.write("%s\t%3.5f\n" % (score_key, _score))
