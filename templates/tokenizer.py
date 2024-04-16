import os
import re
import gc
import json
import math
import pickle
import subprocess
import collections
import unicodedata

from collections import Counter

import numpy
import torch
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader

# >>> {block:tokenizer} <<<

def init_tokenizers(train=False, src_file=None, tgt_file=None, train_data=None, validation_data=None):
    # >>> {segment:tokenizer.create} <<<

    if train:
        # >>> {variable:SRC_VOCAB_SIZE:stmt} <<<
        # >>> {variable:TGT_VOCAB_SIZE:stmt} <<<

        # >>> {segment:tokenizer.training} <<<
        pass

    else:
        src_tokenizer = Tokenizer.load(src_file)
        tgt_tokenizer = Tokenizer.load(tgt_file)

    return src_tokenizer, tgt_tokenizer
