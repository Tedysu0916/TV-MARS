import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
import argparse
from utils.iotools import load_train_configs

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tmars Test")
    parser.add_argument("--config_file", default='logs/Tmars/random_itc+sdm+mlm_rss_ViT-L/14/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    args.test_mode = 'dense'#todo choose your test mode
    if args.test_mode == 'dense':
        args.batch_size = 1
    logger = setup_logger('Tmars', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    val_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, val_loader)