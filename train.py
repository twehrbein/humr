import os
import sys
import torch
import time
import yaml
import argparse
from loguru import logger
from flatten_dict import flatten, unflatten
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar

from train.core.config import update_hparams
from train.core.humr_trainer import HumrTrainer


def train(hparams):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f"Hyperparameters: \n {hparams}")
    experiment_loggers = []
    # initialize tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=hparams.LOG_DIR,
        log_graph=False,
    )
    experiment_loggers.append(tb_logger)

    # load model
    model = HumrTrainer(hparams=hparams)

    ckpt_callback = ModelCheckpoint(
        monitor="train_loss_epoch",
        verbose=True,
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=experiment_loggers,
        log_every_n_steps=100,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        callbacks=[
            ckpt_callback,
            TQDMProgressBar(refresh_rate=hparams.TRAINING.PROG_BAR_FREQ),
        ],
        default_root_dir=hparams.LOG_DIR,
        num_sanity_val_steps=0,
    )

    if args.test:
        trainer.test(model, ckpt_path=hparams.TRAINING.RESUME_CKPT)
    else:
        trainer.fit(model, ckpt_path=hparams.TRAINING.RESUME_CKPT)
    logger.info("*** Finished training ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="cfg file path", required=True)
    parser.add_argument("--log_dir", type=str, help="log dir path", default="./logs")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ckpt", type=str, help="path to checkpoint to load")
    parser.add_argument("--name", help="name of experiment", default="train", type=str)
    args = parser.parse_args()

    hparams = update_hparams(args.cfg)
    if args.name is not None:
        hparams.EXP_NAME = args.name
    if args.ckpt is not None:
        hparams.TRAINING.RESUME_CKPT = args.ckpt
    if args.test:
        hparams.RUN_TEST = True

    # add date to log path
    logtime = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join(
        args.log_dir, logtime + "_" + hparams.EXP_NAME
    )
    os.makedirs(logdir, exist_ok=False)
    hparams.LOG_DIR = logdir
    logger.add(
        os.path.join(hparams.LOG_DIR, "train.log"),
        level="INFO",
        colorize=False,
    )
    command_arguments = " ".join(sys.argv[0:])
    logger.info(f"Command: python {command_arguments}")
    logger.info(f"Input arguments: \n {args}")

    def save_dict_to_yaml(obj, filename, mode="w"):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save final config
    save_dict_to_yaml(
        unflatten(flatten(hparams)), os.path.join(hparams.LOG_DIR, "config_to_run.yaml")
    )

    train(hparams)
