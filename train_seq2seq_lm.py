import pytorch_lightning as pl
from models.seq2seq_lm import argparser
from models.seq2seq_lm.model import Model
from models.seq2seq_lm.data_module import DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from models.seq2seq_lm.config import GPUS, ACCELERATOR
from loguru import logger
from copy import deepcopy
import torch

args = argparser.get_args()

if __name__ == "__main__":
    # load model from_checkpoint or init a new one
    if args.from_checkpoint is None:
        model = Model()
    else:
        print("load from checkpoint")
        model = Model.load_from_checkpoint(args.from_checkpoint)
    # run as a flask api server
    if args.server:
        model.run_server()
        exit()

    # trainer config
    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        # auto_lr_find=Trie,
        # auto_scale_batch_size=True,
        fast_dev_run=args.dev,
        precision=32,
        default_root_dir=args.output_dir,
        max_epochs=args.epoch,
        callbacks=[
            EarlyStopping(monitor="dev_loss", patience=3, mode="min"),
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
        # tuner = pl.tuner.tuning.Tuner(deepcopy(trainer))
        # new_batch_size = tuner.scale_batch_size(
        #     model, datamodule=dm, init_val=torch.cuda.device_count()
        # )
        # del tuner
        # model.hparams.batch_size = new_batch_size
        trainer.fit(model, datamodule=dm)

    # decide which checkpoint to use
    last_model_path = trainer.checkpoint_callback.last_model_path
    best_model_path = trainer.checkpoint_callback.best_model_path
    testing_use_model_path = last_model_path if best_model_path == "" else best_model_path

    logger.info(
        f":testing_use_model_path: {testing_use_model_path}"
    )
    # run_test
    trainer.test(
        model=model if testing_use_model_path == "" else None,
        datamodule=dm,
        ckpt_path=testing_use_model_path,
    )
