import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
from .config import MAX_INPUT_LENGTH, MAX_QUESTION_LENGTH
from utils import ModelEvalMixin
from utils.server import ServerMixin
from utils.scheduler import setup_scheduler, step_scheduler
import wandb

args = get_args()

# Init wandb
# wandb.init(
#     project=args.task_name,
#     name="{0}_{1}_{2}_{3}".format(
#         args.model_name_or_path,
#         args.data_type,
#         str(MAX_INPUT_LENGTH),
#         str(MAX_QUESTION_LENGTH),
#     ),
# )


class Model(pl.LightningModule, ModelEvalMixin, ServerMixin):
    def __init__(self, args=args):
        super().__init__()
        args = get_args()
        self.save_hyperparameters(args)
        self.tokenizer = get_tokenizer(args.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self._type = "seq2seq_lm"

        # wandb.watch(self.model)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @step_scheduler
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1], batch[2])
        loss = outputs["loss"]
        self.log("train_loss", loss, prog_bar=True)
        # wandb.log({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1], batch[2])
        loss = outputs["loss"]
        self.log("dev_loss", loss, prog_bar=True)
        # wandb.log({"dev_loss": loss})

    def test_step(self, batch, batch_idx):
        input_ids = batch[0]
        attention_mask = batch[1]
        ref_question = batch[2][0]
        input_ids_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        assert batch_size == 1

        num_return_sequences = 1
        sample_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_INPUT_LENGTH,
            early_stopping=True,
            temperature=0.85,
            do_sample=True,
            top_p=0.9,
            top_k=10,
            num_beams=3,
            no_repeat_ngram_size=5,
            num_return_sequences=num_return_sequences,
        )

        assert len(sample_outputs) == num_return_sequences  # 1
        sample_output = sample_outputs[0]
        decode_question = self.tokenizer.decode(sample_output, skip_special_tokens=True)
        self.write_predict(decode_question, ref_question)

    def test_epoch_end(self, outputs):
        self.evaluate_predict(dataset=args.data_type)
        self.save_huggingface_model()

    @setup_scheduler
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)
