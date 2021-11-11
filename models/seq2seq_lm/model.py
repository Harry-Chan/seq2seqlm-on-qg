import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
from transformers import get_linear_schedule_with_warmup
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
from utils.scorer import ModelEvalMixin
from loguru import logger

args = get_args()


class Model(pl.LightningModule, ModelEvalMixin):
    def __init__(self, args=args):
        super().__init__()
        args = get_args()
        self.save_hyperparameters(args)
        # self.save_hyperparameters()
        self.tokenizer = get_tokenizer(args.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self._type = "seq2seq_lm"

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1], batch[2])
        loss = outputs["loss"]
        self.log("train_loss", loss)
        self.log("global_step", self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1], batch[2])
        loss = outputs["loss"]
        self.log("dev_loss", loss, prog_bar=True)
        return loss

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
            max_length=args.max_input_length,
            early_stopping=True,
            # temperature=0.85,
            # do_sample=True,
            # top_p=0.9,
            # top_k=10,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token),
            num_beams=args.beam_size,
            no_repeat_ngram_size=5,
            num_return_sequences=num_return_sequences,
        )

        assert len(sample_outputs) == num_return_sequences  # 1
        sample_output = sample_outputs[0]
        decode_question = self.tokenizer.decode(sample_output, skip_special_tokens=True)
        self.write_predict(decode_question, ref_question, args.data_type)

    def test_epoch_end(self, outputs):
        self.evaluate_predict(data_type=args.data_type)
        self.save_huggingface_model()

    def configure_optimizers(self):
        model = self.model
        train_dataloader_size = len(self.train_dataloader())
        num_training_steps = self.trainer.max_epochs * train_dataloader_size
        num_warmup_steps = int(num_training_steps * 0.05)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=1e-8,
        )

        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(
            f"optim scheduler is enable, num_warmup_steps:{num_warmup_steps} num_training_steps:{num_training_steps}"
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
