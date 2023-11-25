import shutil
import torch
from time import  strftime
import os, sys

from sadtalker.aiy import SadTalkerArgs, gen_sad_talker

if __name__ == '__main__':

    # args = SadTalkerArgs('./examples/driven_audio/bus_chinese.wav', './examples/source_image/full_body_1.png', preprocess='full', filename='example')
    args = SadTalkerArgs('./tmp-1/audio.wav', './tmp-1/image.png', preprocess='full', filename='example')

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    gen_sad_talker(args)
