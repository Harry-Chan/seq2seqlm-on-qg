from os import truncate
import pytorch_lightning as pl
from models.seq2seq_lm import argparser
from models.seq2seq_lm.model import Model
from models.seq2seq_lm.data_module import DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models.seq2seq_lm.config import GPUS, ACCELERATOR
import wandb
from loguru import logger

args = argparser.get_args()

if __name__ == "__main__":
    # load model from_checkpoint or init a new one
    if args.from_checkpoint is None:
        model = Model()
    else:
        logger.info(f"load from checkpoint : {args.from_checkpoint}")
        model = Model.load_from_checkpoint(args.from_checkpoint)

    # run as a flask api server
    if args.server:
        model.run_server()
        exit()

    # init wandb
    wandb_logger = WandbLogger(
        project=args.task_name,
        name="{0}_{1}_{2}_{3}".format(
            args.model_name_or_path,
            args.data_type,
            str(args.max_input_length),
            str(args.max_output_length),
        ),
    )

    # trainer config
    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        fast_dev_run=args.dev,
        precision=32,
        default_root_dir=args.output_dir,
        max_epochs=args.epoch,
        logger=wandb_logger,
        log_every_n_steps=args.wandb_logging_steps,
        callbacks=[
            EarlyStopping(monitor="dev_loss", patience=5, mode="min"),
            ModelCheckpoint(
                monitor="dev_loss",
                filename="{epoch}-{dev_loss:.2f}",
                save_top_k=args.epoch,
                mode="min",
            ),
        ],
    )

    # DataModule
    dm = DataModule()

    # train
    if args.run_test == False:
        logger.info(f"Run Training!")
        trainer.fit(model, datamodule=dm)

    if args.dev == 0:
        # decide which checkpoint to use
        last_model_path = trainer.checkpoint_callback.last_model_path
        best_model_path = trainer.checkpoint_callback.best_model_path
        testing_use_model_path = (
            last_model_path if best_model_path == "" else best_model_path
        )

        logger.info(f"Run Testing!")
        if args.run_test == True:
            logger.info(f"testing_use_model_path: {args.from_checkpoint}")
        else:
            logger.info(f"testing_use_model_path: {testing_use_model_path}")

        # run_test
        trainer.test(
            model=model if testing_use_model_path == "" else None,
            datamodule=dm,
            ckpt_path=testing_use_model_path,
        )
        
    wandb.finish()