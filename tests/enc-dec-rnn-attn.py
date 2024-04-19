import os
import math

import numpy
import torch
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

        model = self.module.RNNEncoderDecoderLMWithAttention(**used_params)

        criterion = self.module.get_criterion(model, self.src_tokenizer, self.tgt_tokenizer)
        if std:
            optimizer = torch.optim.AdamW(model.parameters())
        else:
            optimizer = self.module.get_optimizer(model, self.src_tokenizer, self.tgt_tokenizer)

        trainer = self.module.RNNEncoderDecoderTrainer("rnn-attn", model, criterion, optimizer)

        train_dataset = TokenizerDataset(self.train_data, self.src_tokenizer, self.tgt_tokenizer, **used_data_params)

        rnn_enc_dec_train_data = dict(
            train_dataset=train_dataset,
            collate_fn=train_dataset.collate
        )

        trainer.train(**rnn_enc_dec_train_data, **used_train_params)

        return model, trainer

    def load_components(self, model=True):
        load_tokenizers = getattr(self.module, 'load_tokenizers')
        load_model      = getattr(self.module, 'load_model')

        src_file = self.record.deepget(("files", "found", "src-tokenizer/tokenizer.pkl"))
        tgt_file = self.record.deepget(("files", "found", "tgt-tokenizer/tokenizer.pkl"))
        mdl_file = self.record.deepget(("files", "found", "rnn.enc-dec.attn/model.pt"))

        root = self.record.deepget("meta.root")

        if self.src_tokenizer is None and self.tgt_tokenizer is None:
            if root is None or src_file is None or tgt_file is None:
                raise ValueError("Tokenizer file(s) not found")

            self.src_tokenizer, self.tgt_tokenizer = load_tokenizers(
                os.path.join(root, "src-tokenizer"),
                os.path.join(root, "tgt-tokenizer"),
                self.train_data, self.validation_data
            )

        if model and self.model is None:
            if root is None or mdl_file is None:
                raise ValueError("Model file not found")

            try:
                self.model = load_model(os.path.join(root, "rnn.enc-dec.attn"), 'cuda:0')
            except:
                self.model = load_model(os.path.join(root, "rnn.enc-dec.attn"), 'cpu')

    def logprobs(self, seq_x, seq_y):
        assert self.model is not None and self.src_tokenizer is not None

        self.model.eval()

        with torch.no_grad():
            seq_x = self.src_tokenizer.encode(seq_x, add_start=True, add_end=True)
            seq_x = torch.tensor(seq_x, device=self.model.device)

            seq_y = self.tgt_tokenizer.encode(seq_y, add_start=True, add_end=True)
            seq_y = torch.tensor(seq_y, device=self.model.device)

            return self.model.log_probability(seq_x, seq_y)

def test_model_functions_correctness(context: Context):
    if context.model is None: context.load_components()

    seq_x = torch.tensor(context.src_tokenizer.encode("aamish"), device=context.model.device)
    seq_y = context.tgt_tokenizer.encode("आमिष", add_start=True)[:1]
    seq_y = torch.tensor(seq_y, device=context.model.device)

    try:
        output, _ = context.model(seq_x, seq_y)
        output, _, attn = context.model(seq_x, seq_y, output_attention=True)

        assert math.isclose(torch.exp(output).sum(dim=-1).squeeze().item(), 1.0, rel_tol=1e-4)
        assert math.isclose(attn.sum(dim=-1).squeeze().item(), 1.0)
    except AssertionError:
        raise AssertionError("forward not executable or return mismatch")
    except Exception:
        try:
            seq_x, seq_y = torch.stack([ seq_x ], dim=0), torch.stack([ seq_y ], dim=0)
            output, _ = context.model(seq_x, seq_y)
            output, _, attn = context.model(seq_x, seq_y, output_attention=True)

            assert math.isclose(torch.exp(output).sum(dim=-1).squeeze().item(), 1.0, rel_tol=1e-4)
            assert math.isclose(attn.sum(dim=-1).squeeze().item(), 1.0)
        except:
            raise AssertionError("forward not executable or return mismatch")

def test_model_attentions(context: Context):
    if context.model is None: context.load_components()

    train_subset = context.train_data.sample(n=5, random_state=20240401)
    terminate_token = context.tgt_tokenizer.encode("आमिष", add_start=True)[-1]

    for name in train_subset['Name']:
        gen_translation = context.module.rnn_greedy_generate(
            context.model, name, context.src_tokenizer,
            context.tgt_tokenizer, max_length=30
        )
        seq_x = torch.tensor(context.src_tokenizer.encode(name), device=context.model.device)

        try:
            attentions, gen_translation_attn = context.model.attentions(
                seq_x, terminate_token, max_length=30
            )
        except:
            attentions, gen_translation_attn = context.model.attentions(
                torch.stack([ seq_x ], dim=0), terminate_token, max_length=30
            )

        gen_translation_attn = gen_translation_attn.squeeze().cpu().to(dtype=torch.long).tolist()
        gen_translation_attn = context.tgt_tokenizer.decode(gen_translation_attn, strip_special=True)
        assert gen_translation == gen_translation_attn, "assert fail: decoding strategy not greedy"
        attentions = attentions.squeeze().sum(dim=-1).cpu()
        assert torch.allclose(attentions, torch.ones_like(attentions)), "assert fail: attention scores not normalized"

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

    pt_scores = context.record.deepget("outputs.enc-dec-rnn-attn.pretrained")

    if pt_scores is None:
        pt_scores = evaluator.evaluate(
            context.model, test_subset['Name'], test_subset['Translation'],
            max_length = 100
        )

    rt_scores  = context.record.deepget("outputs.enc-dec-rnn-attn.retrained")
    rt_outputs = context.record.deepget("outputs.enc-dec-rnn-attn.retrained_val_outputs")

    output_file = context.record.deepget(("files", "found", "rnn.enc-dec.attn/outputs.csv"))

    if output_file is None:
        raise AssertionError("pretrained output file not found")

    root = context.record.deepget("meta.root")
    given_pt_outputs = pandas.read_csv(os.path.join(root, "rnn.enc-dec.attn", "outputs.csv"))['Translation'].tolist()

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

    vanilla_scores = { 'bleu': 0.55, 'accuracy': 0.15, 'cer': 0.3, 'ter': 0.45 }
    vanilla_scores = context.record.deepget("outputs.enc-dec-rnn.trained", vanilla_scores)

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

    try:
        assert scores['bleu'] > min(0.55, vanilla_scores['bleu'])
        assert scores['accuracy'] > min(0.15, vanilla_scores['accuracy'])
        assert scores['cer'] < max(0.3, vanilla_scores['cer'])
        assert scores['ter'] < max(0.45, vanilla_scores['ter'])

        return { 'trained': scores }
    except AssertionError as exc:
        setattr(exc, 'outputs', { 'trained': scores })
        raise exc

def test_model_quality_best_hyperparams(context: Context):
    if context.model is None: context.load_components()

    vanilla_scores = { 'bleu': 0.55, 'accuracy': 0.15, 'cer': 0.3, 'ter': 0.45 }
    vanilla_scores = context.record.deepget("outputs.enc-dec-rnn.pretrained", vanilla_scores)
    scores = context.record.deepget("outputs.enc-dec-rnn-attn.pretrained")

    if scores is None:
        if context.model is None: context.load_components()

        evaluator = Evaluator(context.src_tokenizer, context.tgt_tokenizer)
        evaluator.set_decoding_method(context.module.rnn_greedy_generate)

        test_subset = context.test_data

        scores = evaluator.evaluate(
            context.model, test_subset['Name'], test_subset['Translation'],
            max_length = 100
        )

    assert scores['bleu'] > min(0.55, vanilla_scores['bleu'])
    assert scores['accuracy'] > min(0.15, vanilla_scores['accuracy'])
    assert scores['cer'] < max(0.3, vanilla_scores['cer'])
    assert scores['ter'] < max(0.45, vanilla_scores['ter'])