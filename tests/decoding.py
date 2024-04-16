import os
import math
import inspect

import torch

from .common import Evaluator, read_dataframe

class Context:
    def __init__(self, module, record, data_dir) -> None:
        self.module = module
        self.record = record
        self.data_dir = data_dir

        self.attempted_bonus = self.record.deepget("variables.ATTEMPTED_BONUS", "False").lower() == "true"

        self.train_data      = read_dataframe("train", os.path.join(data_dir, "data"))
        self.validation_data = read_dataframe("valid", os.path.join(data_dir, "data"))
        self.test_data       = read_dataframe("test", os.path.join(data_dir, "data"))

        self.model = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None

    def load_components(self):

        load_tokenizers = getattr(self.module, 'load_tokenizers')
        load_model      = getattr(self.module, 'load_model')

        src_file = self.record.deepget(("files", "found", "src-tokenizer/tokenizer.pkl"))
        tgt_file = self.record.deepget(("files", "found", "tgt-tokenizer/tokenizer.pkl"))
        mdl_file = self.record.deepget(("files", "found", "rnn.enc-dec/model.pt"))

        root = self.record.deepget("meta.root")

        if root is None or src_file is None or tgt_file is None:
            raise ValueError("Tokenizer file(s) not found")

        if root is None or mdl_file is None:
            raise ValueError("Model file not found")

        self.src_tokenizer, self.tgt_tokenizer = load_tokenizers(
            os.path.join(root, "src-tokenizer"),
            os.path.join(root, "tgt-tokenizer"),
            self.train_data, self.validation_data
        )

        try:
            self.model = load_model(os.path.join(root, "rnn.enc-dec"), 'cuda:0')
        except:
            self.model = load_model(os.path.join(root, "rnn.enc-dec"), 'cpu')

    def logprobs(self, seq_x, seq_y):
        assert self.model is not None and self.src_tokenizer is not None

        self.model.eval()

        with torch.no_grad():
            seq_x = self.src_tokenizer.encode(seq_x, add_start=True, add_end=True)
            seq_x = torch.tensor(seq_x, device=self.model.device)

            seq_y = self.tgt_tokenizer.encode(seq_y, add_start=True, add_end=True)
            seq_y = torch.tensor(seq_y, device=self.model.device)

            return self.model.log_probability(seq_x, seq_y)

def test_attempted_bonus_decoding(context: Context):
    assert context.attempted_bonus, "Student did not attempt the bonus module"

def test_decoding_correctness_beam(context: Context):
    assert context.attempted_bonus, "Student did not attempt the bonus module"

    signature = inspect.signature(context.module.rnn_better_generate)
    looks_like_beam_search = 'k' in signature.parameters

    if looks_like_beam_search:

        if context.model is None: context.load_components()

        evaluator = Evaluator(context.src_tokenizer, context.tgt_tokenizer)
        evaluator.set_decoding_method(context.module.rnn_greedy_generate)

        test_subset = context.test_data.sample(n=10, random_state=20240401)

        decoding_params = { **getattr(context.module, 'decoding_params', {}) }
        decoding_params.update({ 'k': 1, 'length_penalty_alpha': 0 })

        if 'max_length' in decoding_params:
            decoding_params = { **decoding_params }
            del decoding_params['max_length']

        greedy_scores = evaluator.evaluate(
            context.model, test_subset['Name'], test_subset['Translation'],
            max_length = 100
        )

        evaluator.set_decoding_method(context.module.rnn_better_generate)

        better_scores = evaluator.evaluate(
            context.model, test_subset['Name'], test_subset['Translation'],
            max_length = 100, **decoding_params
        )

        assert all(
            math.isclose(greedy_scores[metric], better_scores[metric])
            for metric in greedy_scores
        ), \
            "assertion fail: beam search with k=1 not equal to greedy"

def test_decoding_correctness_probs(context: Context):
    assert context.attempted_bonus, "Student did not attempt the bonus module"

    decoding_params = getattr(context.module, 'decoding_params', {})

    if 'max_length' in decoding_params:
        decoding_params = { **decoding_params }
        del decoding_params['max_length']

    if context.model is None: context.load_components()

    names = context.test_data['Name'].sample(n=5, random_state=20240401)

    for name in names:
        greedy_seq_y = context.module.rnn_greedy_generate(
            context.model, name, context.src_tokenizer, context.tgt_tokenizer,
            max_length=100
        )
        better_seq_y = context.module.rnn_better_generate(
            context.model, name, context.src_tokenizer, context.tgt_tokenizer,
            max_length=100, **decoding_params
        )

        greedy_logprob = context.logprobs(name, greedy_seq_y)
        better_logprob = context.logprobs(name, better_seq_y)

        assert better_logprob >= greedy_logprob, \
            "assertion fail: log probability does not match expected"

def test_decoding_performance(context: Context):
    assert context.attempted_bonus, "Student did not attempt the bonus module"

    if context.model is None: context.load_components()

    evaluator = Evaluator(context.src_tokenizer, context.tgt_tokenizer)
    evaluator.set_decoding_method(context.module.rnn_greedy_generate)

    test_subset = context.test_data.sample(n=256, random_state=20240401)

    decoding_params = getattr(context.module, 'decoding_params', {})

    if 'max_length' in decoding_params:
        decoding_params = { **decoding_params }
        del decoding_params['max_length']

    greedy_scores = evaluator.evaluate(
        context.model, test_subset['Name'], test_subset['Translation'],
        max_length = 100
    )

    evaluator.set_decoding_method(context.module.rnn_better_generate)

    better_scores = evaluator.evaluate(
        context.model, test_subset['Name'], test_subset['Translation'],
        max_length = 100, **decoding_params
    )

    assert greedy_scores['bleu'] <= better_scores['bleu'], \
        "assertion fail: better generate not better than greedy"
    assert greedy_scores['accuracy'] <= better_scores['accuracy'], \
        "assertion fail: better generate not better than greedy"
    assert greedy_scores['cer'] >= better_scores['cer'], \
        "assertion fail: better generate not better than greedy"
    assert greedy_scores['ter'] >= better_scores['ter'], \
        "assertion fail: better generate not better than greedy"

    return {
        'scores': {
            'greedy': greedy_scores,
            'better': better_scores
        }
    }
