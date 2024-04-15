import os
import json

import torch
import numpy
import pandas
from torch.utils.data import TensorDataset

from .common import read_dataframe, Evaluator

# ==== Test suite specific utilities

class TokenizerDataset(TensorDataset):
    """ Abstraction of the tokenizer functions as a pytorch dataset. """

    def __init__(self, data, src_tokenizer, tgt_tokenizer, src_padding=None, tgt_padding=None):
        """ Initializes the dataset.

        Args:
            data: DataFrame of input and output strings.
            src_tokenizer (Tokenizer): Tokenizer for the source language.
            tgt_tokenizer (Tokenizer): Tokenizer for the target language.
            src_padding (int, optional): Padding length for the source text. Defaults to None.
            tgt_padding (int, optional): Padding length for the target text. Defaults to None.
        """

        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_padding = src_padding
        self.tgt_padding = tgt_padding

    def collate(self, batch):
        """ Collates data instances into a batch of tokenized tensors.

        Args:
            batch (list[tuple]): List of x, y pairs.

        Returns:
            tuple[torch.Tensor|PackedSequence, torch.Tensor|PackedSequence]: pair of tokenized tensors.
        """

        x_batch = [ data[0] for data in batch ]
        y_batch = [ data[1] for data in batch ]

        x_batch = self.src_tokenizer.batch_encode(x_batch, self.src_padding)
        y_batch = self.tgt_tokenizer.batch_encode(y_batch, self.tgt_padding)

        if self.src_padding is None:
            x_batch = torch.nn.utils.rnn.pack_sequence([ torch.tensor(tokens) for tokens in x_batch ], False)
        else:
            x_batch = torch.tensor(x_batch)

        if self.tgt_padding is None:
            y_batch = torch.nn.utils.rnn.pack_sequence([ torch.tensor(tokens) for tokens in y_batch ], False)
        else:
            y_batch = torch.tensor(y_batch)

        return x_batch, y_batch

    def __getitem__(self, index):
        """ Returns the nth instance from the dataset.

        Args:
            index (int): Index of the instance to retrieve.

        Returns:
            tuple[str, str]: Untokenized instance pair.
        """

        return (
            self.data['Name'][index],
            self.data['Translation'][index]
        )

    def __len__(self):
        """ Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

# ==== Tests

class Context:
    def __init__(self, module, record, data_dir) -> None:
        self.module = module
        self.record = record
        self.data_dir = data_dir

        self.train_data      = read_dataframe("train", os.path.join(data_dir, "data"))
        self.validation_data = read_dataframe("valid", os.path.join(data_dir, "data"))
        self.test_data       = read_dataframe("test", os.path.join(data_dir, "data"))

        self.model = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.trained_model = None

    def load_components(self, model=True):
        load_tokenizers = getattr(self.module, 'load_tokenizers')
        load_model      = getattr(self.module, 'load_model')

        src_file = self.record.deepget(("files", "found", "src-tokenizer/tokenizer.pkl"))
        tgt_file = self.record.deepget(("files", "found", "tgt-tokenizer/tokenizer.pkl"))
        mdl_file = self.record.deepget(("files", "found", "rnn.enc-dec/model.pt"))

        root = self.record.deepget("meta.root")

        if self.src_tokenizer is None and self.tgt_tokenizer is None:
            if root is None or src_file is None or tgt_file is None:
                raise ValueError("Tokenizer file(s) not found")

            if root is None or mdl_file is None:
                raise ValueError("Model file not found")

            self.src_tokenizer, self.tgt_tokenizer = load_tokenizers(
                os.path.join(root, "src-tokenizer"),
                os.path.join(root, "tgt-tokenizer")
            )

        if model and self.model is None:
            try:
                self.model = load_model(os.path.join(root, "rnn.enc-dec"), 'cuda:0')
            except:
                self.model = load_model(os.path.join(root, "rnn.enc-dec"), 'cpu')

    def train_model(self, params=None, data_params=None, train_params=None, std=False):
        used_params, used_data_params, used_train_params = self.module.load_params(
            self.src_tokenizer, self.tgt_tokenizer
        )

        for param in (params or {}):
            if param in used_params: used_params[param] = params[param]

        for param in (data_params or {}):
            if param in used_data_params: used_data_params[param] = data_params[param]

        for param in (train_params or {}):
            if param in used_train_params: used_train_params[param] = train_params[param]

        if 'save_steps' in used_train_params: del used_train_params['save_steps']
        used_train_params['eval_steps'] = 100000

        torch.manual_seed(42)

        model = self.module.RNNEncoderDecoderLM(**used_params)

        criterion = self.module.get_criterion(model)
        optimizer = torch.optim.AdamW(model.parameters()) if std else self.module.get_optimizer(model)

        trainer = self.module.RNNEncoderDecoderTrainer("rnn", model, criterion, optimizer)

        train_dataset = TokenizerDataset(self.train_data, self.src_tokenizer, self.tgt_tokenizer, **used_data_params)

        rnn_enc_dec_train_data = dict(
            train_dataset=train_dataset,
            collate_fn=train_dataset.collate
        )

        trainer.train(**rnn_enc_dec_train_data, **used_train_params)

        return model, trainer

    def logprobs(self, seq_x, seq_y):
        assert self.model is not None and self.src_tokenizer is not None

        self.model.eval()

        with torch.no_grad():
            seq_x = self.src_tokenizer.encode(seq_x, add_start=True, add_end=True)
            seq_x = torch.tensor(seq_x, device=self.model.device)

            seq_y = self.tgt_tokenizer.encode(seq_y, add_start=True, add_end=True)
            seq_y = torch.tensor(seq_y, device=self.model.device)

            try:
                return self.model.log_probability(seq_x, seq_y)
            except:
                print(torch.stack([ seq_x ], dim=0).shape)
                return self.model.log_probability(torch.stack([ seq_x ], dim=0), torch.stack([ seq_y ], dim=0))

    def greedy_generate(self, seq_x, max_length):
        self.model.eval()

        with torch.no_grad():
            seq_x = torch.tensor(self.src_tokenizer.encode(seq_x)).to(self.model.device)
            dummy_seq = self.tgt_tokenizer.encode("आमिष", add_start=True, add_end=True)[:1]
            seq_y, end_token = [ dummy_seq[0] ], dummy_seq[-1]

            try:
                decoder_hidden_state = None

                for _ in range(max_length):
                    logits, decoder_hidden_state = self.model(
                        seq_x, torch.tensor(seq_y[-1:], device=self.model.device),
                        decoder_hidden_state
                    )
                    seq_y.append(logits[-1].cpu().argmax().item())
                    if seq_y[-1] == end_token: break

                return self.tgt_tokenizer.decode(seq_y)

            except:
                seq_x = torch.stack([ seq_x ], dim=0)
                seq_y, end_token = [ dummy_seq[0] ], dummy_seq[-1]
                decoder_hidden_state = None

                for _ in range(max_length):
                    logits, decoder_hidden_state = self.model(
                        seq_x, torch.tensor([ seq_y[-1:] ], device=self.model.device),
                        decoder_hidden_state
                    )
                    seq_y.append(logits[:, -1].cpu().argmax().item())
                    if seq_y[-1] == end_token: break

                return self.tgt_tokenizer.decode(seq_y)

def test_model_functions_correctness(context: Context):
    if context.model is None: context.load_components()

    seq_x = torch.tensor(context.src_tokenizer.encode("aamish"), device=context.model.device)
    seq_y = context.tgt_tokenizer.encode("आमिष", add_start=True)[:1]
    seq_y = torch.tensor(seq_y, device=context.model.device)

    try:
        output, _ = context.model(seq_x, seq_y)
    except:
        try:
            output, _ = context.model(torch.stack([ seq_x ], dim=0), torch.stack([ seq_y ], dim=0))
        except:
            raise AssertionError("forward not executable or return mismatch")

def test_model_functions_adherence(context: Context):
    if context.model is None: context.load_components()

    # Test for single input

    seq_x = torch.tensor(context.src_tokenizer.encode("aamish"), device=context.model.device)
    seq_y = context.tgt_tokenizer.encode("आमिष", add_start=True)[:1]
    seq_y = torch.tensor(seq_y, device=context.model.device)

    try:
        output, _ = context.model(seq_x, seq_y)
        assert 2 <= len(output.shape) <= 3
        if len(output.shape) == 3: output = output[0]
        assert output.shape[0] == 1
        assert -1 <= (output.shape[-1] - len(context.tgt_tokenizer.get_vocabulary())) <= 1
    except AssertionError:
        raise AssertionError("assert fail: output dim mismatch")
    except Exception:
        output, _ = context.model(torch.cat([seq_x], dim=0), torch.cat([seq_y], dim=0))
        assert 2 <= len(output.shape) <= 3
        if len(output.shape) == 3: output = output[0]
        assert output.shape[0] == 1
        assert -1 <= (output.shape[-1] - len(context.tgt_tokenizer.get_vocabulary())) <= 1

    # Test for batched input

    seq_x = torch.stack([ seq_x, seq_x, seq_x ], dim=0)
    seq_y = torch.stack([ seq_y, seq_y, seq_y ], dim=0)

    output, _ = context.model(seq_x, seq_y)

    assert len(output.shape) == 3
    assert output.shape[0]   == 3
    assert output.shape[1]   == 1
    assert -1 <= (output.shape[-1] - len(context.tgt_tokenizer.get_vocabulary())) <= 1

def test_model_logprobs(context: Context):
    if context.model is None: context.load_components()

    train_subset = context.train_data.sample(n=5, random_state=20240401)
    test_subset  = context.test_data.sample(n=5, random_state=20240401)

    for name, translation in zip(train_subset['Name'], train_subset['Translation']):
        logprob = context.logprobs(name, translation)
        assert logprob <= 0
        assert logprob >= -2

    for name, translation in zip(test_subset['Name'], test_subset['Translation']):
        logprob = context.logprobs(name, translation)
        assert logprob <= 0
        gen_translation = context.module.rnn_greedy_generate(
            context.model, name, context.src_tokenizer,
            context.tgt_tokenizer, max_length=30
        )
        gen_logprob = context.logprobs(name, gen_translation)
        assert -10 <= (gen_logprob - logprob) <= 10

def test_model_greedy_generate(context: Context):
    if context.model is None: context.load_components()

    test_subset = context.test_data.sample(n=5, random_state=20240401)

    for name in test_subset['Name']:
        assumed_seq_y = context.greedy_generate(name, 100)

        context.model.eval()
        with torch.no_grad():
            predicted_seq_y = context.module.rnn_greedy_generate(
                context.model, name, context.src_tokenizer,
                context.tgt_tokenizer, max_length=100
            )

        assert all(tok_1 == tok_2 for tok_1, tok_2 in zip(assumed_seq_y, predicted_seq_y)), \
            "assert fail: greedy decoding not as expected"

def test_model_training(context: Context):
    if context.src_tokenizer is None: context.load_components(model=False)

    model, trainer = context.train_model()

    test_subset = context.test_data
    evaluator = Evaluator(context.src_tokenizer, context.tgt_tokenizer)
    evaluator.set_decoding_method(context.module.rnn_greedy_generate)

    scores = evaluator.evaluate(
        model, test_subset['Name'], test_subset['Translation'],
        max_length = 100
    )

    val_outputs = []
    _, data_params, _ = context.module.load_params(context.src_tokenizer, context.tgt_tokenizer)

    model.eval()

    with torch.no_grad():
        for name in context.validation_data['Name']:
            val_outputs.append(context.module.rnn_greedy_generate(
                model, name, context.src_tokenizer,
                context.tgt_tokenizer, max_length=data_params.get('tgt_padding', 100)
            ))

    context.trained_model = (model, trainer)

    return {
        'retrained': scores,
        'retrained_val_outputs': val_outputs,
        'retrained_loss': trainer.loss_history['train'],
    }

def test_model_consistency(context: Context):
    if context.model is None: context.load_components()

    evaluator = Evaluator(context.src_tokenizer, context.tgt_tokenizer)
    evaluator.set_decoding_method(context.module.rnn_greedy_generate)

    test_subset = context.test_data

    pt_scores = context.record.deepget("outputs.enc-dec-rnn.pretrained")

    if pt_scores is None:
        pt_scores = evaluator.evaluate(
            context.model, test_subset['Name'], test_subset['Translation'],
            max_length = 100
        )

    rt_scores  = context.record.deepget("outputs.enc-dec-rnn.retrained")
    rt_outputs = context.record.deepget("outputs.enc-dec-rnn.retrained_val_outputs")

    output_file = context.record.deepget(("files", "found", "rnn.enc-dec/outputs.csv"))

    if output_file is None:
        raise AssertionError("pretrained output file not found")

    root = context.record.deepget("meta.root")
    given_pt_outputs = pandas.read_csv(os.path.join(root, "rnn.enc-dec", "outputs.csv"))['Translation'].tolist()

    pt_outputs = []
    _, data_params, _ = context.module.load_params(context.src_tokenizer, context.tgt_tokenizer)

    context.model.eval()

    with torch.no_grad():
        for name in context.validation_data['Name']:
            pt_outputs.append(context.module.rnn_greedy_generate(
                context.model, name, context.src_tokenizer,
                context.tgt_tokenizer, max_length=data_params.get('tgt_padding', 100)
            ))

    if rt_scores is None or rt_outputs is None:
        model, _ = context.train_model()

        rt_scores = evaluator.evaluate(
            model, test_subset['Name'], test_subset['Translation'],
            max_length = 100
        )

        rt_outputs = []

        model.eval()

        with torch.no_grad():
            for name in context.validation_data['Name']:
                rt_outputs.append(context.module.rnn_greedy_generate(
                    model, name, context.src_tokenizer,
                    context.tgt_tokenizer, max_length=data_params.get('tgt_padding', 100)
                ))

    assert all(rt_o == pt_o for rt_o, pt_o in zip(given_pt_outputs, pt_outputs)),\
        "assert fail: output mismatch (pretrained)"
    assert len([rt_o == pt_o for rt_o, pt_o in zip(rt_outputs, pt_outputs)]) > (0.6 * len(pt_outputs)),\
        "assert fail: output mismatch (retrained)"
    assert all(numpy.isclose(rt_scores[metric], pt_scores[metric], atol=1e-2, rtol=1e-2) for metric in rt_scores),\
        "assert fail: score mismatch"

def test_model_performance(context: Context):
    if context.model is None: context.load_components()

    evaluator = Evaluator(context.src_tokenizer, context.tgt_tokenizer)
    evaluator.set_decoding_method(context.module.rnn_greedy_generate)

    test_subset = context.test_data

    scores = evaluator.evaluate(
        context.model, test_subset['Name'], test_subset['Translation'],
        max_length = 100
    )

    return { 'pretrained': scores }

def test_model_quality_fixed_hyperparams(context: Context):
    if context.model is None: context.load_components(model=False)

    params = {
        'embd_dims'  : 256,
        'hidden_size': 512,
        'dropout'    : 0,
        'num_layers' : 1
    }
    train_params = dict(
        num_epochs=10,
        batch_size=16,
        shuffle=True,
    )

    model, _ = context.train_model(params, None, train_params, True)

    scores = context.record.deepget("outputs.enc-dec-rnn.trained")

    evaluator = Evaluator(context.src_tokenizer, context.tgt_tokenizer)
    evaluator.set_decoding_method(context.module.rnn_greedy_generate)

    test_subset = context.test_data

    scores = evaluator.evaluate(
        model, test_subset['Name'], test_subset['Translation'],
        max_length = 100
    )

    assert scores['bleu'] > 0.55
    assert scores['accuracy'] > 0.15
    assert scores['cer'] < 0.3
    assert scores['ter'] < 0.45

    return { 'trained': scores }

def test_model_quality_best_hyperparams(context: Context):

    scores = context.record.deepget("outputs.enc-dec-rnn.pretrained")

    if scores is None:
        if context.model is None: context.load_components()

        evaluator = Evaluator(context.src_tokenizer, context.tgt_tokenizer)
        evaluator.set_decoding_method(context.module.rnn_greedy_generate)

        test_subset = context.test_data

        scores = evaluator.evaluate(
            context.model, test_subset['Name'], test_subset['Translation'],
            max_length = 100
        )

    assert scores['bleu'] > 0.55
    assert scores['accuracy'] > 0.15
    assert scores['cer'] < 0.3
    assert scores['ter'] < 0.45