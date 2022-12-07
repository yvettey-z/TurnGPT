from argparse import ArgumentParser

import pytorch_lightning as pl

from datasets_turntaking.dialog_text_dm import ConversationalDM
from turngpt.model import TurnGPT

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

PROJECT = "TurnGPT"
SAVE_DIR = "runs/TurnGPT"


def test():
    parser = ArgumentParser()
    parser = TurnGPT.add_model_specific_args(parser)
    parser = ConversationalDM.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name_info", type=str, default="")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--load_from_checkpoint", action='store_true')
    parser.add_argument("--resume", default=None)
    parser.add_argument("--id", default=None)
    args = parser.parse_args()

    print("Datasets: ", args.datasets)

    pl.seed_everything(args.seed)

    # Model
    print("Loading Model...")
    ckpt_path = None
    if args.load_from_checkpoint:
        model = TurnGPT.load_from_checkpoint(args.pretrained_model_name_or_path)
        ckpt_path = args.pretrained_model_name_or_path
    else:
        raise ValueError("You must have the --load_from_checkpoint flag for testing")
    model.print_parameters()

    # DataModule
    dm = ConversationalDM(
        datasets=args.datasets,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        savepath=args.savepath,
        overwrite=args.overwrite,
        load_from_cache_file=args.load_from_cache_file,
        num_proc=args.num_proc,
    )
    dm.prepare_data()
    dm.setup("test")

    # Callbacks & Logger
    callbacks = None

    # Trainer
    args.devices = 1
    trainer = pl.Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
    )

    trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    test()
