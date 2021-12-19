import os
import yaml
import argparse
from easydict import EasyDict as edict
from utils import Trainer
import time
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('act',
                        choices=["train","transform"],
                        default="train", type=str)
    parser.add_argument('--config',
                        help='configuration filename',
                        default="./configs/base_config.yml", type=str)
    parser.add_argument('--epoch',
                        help='epoch',
                        default=None, type=int)
    return parser.parse_args()

args = parse_args()
if args.config is None:
    raise Exception("no configuration file.")
with open(args.config, 'r') as fid:
    config = edict(yaml.load(fid, Loader=yaml.FullLoader))
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
# path of checkpoint
config.checkpoint = os.path.join(config.WORK_PATH, "checkpoint_"+re.search(r"[/]?([^/]+)\.yml", args.config).group(1))
# init trainer
trainer = Trainer(config)
config.act = args.act
if args.act == "train":
    trainer.fit()
elif args.act == "eval":
    trainer.load_checkpoint(args.epoch)
    trainer.eval()
else:
    trainer.transform(args.epoch)
