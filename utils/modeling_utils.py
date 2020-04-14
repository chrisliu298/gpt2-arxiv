from utils.data import TextDataset, LineByLineTextDataset

import glob
import logging
import os
import random
from typing import List
import numpy as np
import torch

logger = logging.getLogger(__name__)

def sorted_checkpoints(
    args, checkpoint_prefix="checkpoint", use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix))
    )

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))

            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer, args, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer, args, file_path=file_path, block_size=args.block_size
        )
