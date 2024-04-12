import os
import re
import gc
import json
import math
import pickle
import subprocess
import collections
import unicodedata

import numpy
import torch
import pandas as pd
import tqdm.auto as tqdm
import torch.nn.functional as F

# >>> {block:tokenizer} <<<

# >>> {block:enc-dec-rnn} <<<

def rnn_greedy_generate(model, seq_x, src_tokenizer, tgt_tokenizer, max_length):
    """ Given a source string, translate it to the target language using the trained model.
        This function should perform greedy sampling to generate the results.

    Args:
        model (nn.Module): RNN Type Encoder-Decoder Model
        seq_x (str): Input string to translate.
        src_tokenizer (Tokenizer): Source language tokenizer.
        tgt_tokenizer (Tokenizer): Target language tokenizer.
        max_length (int): Maximum length of the target sequence to decode.

    Returns:
        str: Generated string for the given input in the target language.
    """

    # >>> {segment:enc-dec-rnn.greedy_generate} <<<
    pass

# >>> {block:enc-dec-rnn.better_generate} <<<

# >>> {segment:decoding.init} <<<

def load_tokenizers(src_file=None, tgt_file=None, train_data=None, validation_data=None):
    # >>> {segment:tokenizer.create} <<<

    src_tokenizer = Tokenizer.load(src_file)
    tgt_tokenizer = Tokenizer.load(tgt_file)

    return src_tokenizer, tgt_tokenizer

def load_model(model_path, device):
    model = torch.load(os.path.join(model_path, "model.pt"), map_location=device)
    return model.to(device)