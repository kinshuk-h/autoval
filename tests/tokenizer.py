import os
import math
import collections
import unicodedata

import torch
import tqdm.auto as tqdm
from torch.utils.data import TensorDataset, DataLoader

from .common import read_dataframe, Evaluator

# ==== Test suite specific utilities

class BPETokenizer:
    """ Represents the tokenizer for text data.
        Provides methods to encode and decode strings (as instance or as a batch). """

    def __init__(self):
        """ Initializes a new tokenizer.

            Any variables required in intermediate operations are declared here.
            You will also need to define things like special tokens and other things here.

            All variables declared in this function will be serialized
                and deserialized when loading and saving the Tokenizer.
            """

        # BEGIN CODE : tokenizer.init

        # ADD YOUR CODE HERE
        self.special_tokens = { '[BOS]': 1, '[EOS]': 2, '[PAD]': 0 }
        self.vocab = { bytes([ i ]): i+len(self.special_tokens) for i in range(256)  }
        self.merge_rules = {  }
        self.inv_vocab = { _id: token for token, _id in self.vocab.items() }
        self.inv_vocab.update({ _id: token.encode() for token, _id in self.special_tokens.items() })

        # END CODE

    def train(self, data, vocab_size):
        """ Trains a tokenizer to learn meaningful representations from input data.
            In the end, learns a vocabulary of a fixed size over the given data.
            Special tokens, if any, must not be counted towards this vocabulary.

        Args:
            data (list[str]): List of input strings from a text corpus.
            vocab_size (int): Final desired size of the vocab to be learnt.
        """

        # BEGIN CODE : tokenizer.train

        # ADD YOUR CODE HERE
        self.vocab = { bytes([ i ]): i+len(self.special_tokens) for i in range(256)  }
        self.vocab.update({ token.encode('utf-8'): _id for token, _id in self.special_tokens.items() })

        self.merge_rules = {  }
        self.inv_vocab   = { _id: token for token, _id in self.vocab.items() }

        data = [ [ i+len(self.special_tokens) for i in instance.encode('utf-8') ] for instance in data ]

        while len(self.vocab) < len(self.special_tokens) + vocab_size:
            # Compute stats
            counts = collections.defaultdict(int)
            for tok_str in data:
                for tok, next_tok in zip(tok_str, tok_str[1:]):
                    counts[(tok, next_tok)] += 1

            # Learn a new merge rule
            best_pair = max(counts, key=counts.get)
            new_token, new_id = self.inv_vocab[best_pair[0]] + self.inv_vocab[best_pair[1]], len(self.vocab) + 1
            self.merge_rules[best_pair] = new_id
            self.inv_vocab[new_id] = new_token
            self.vocab[new_token]  = new_id

            # Update tokens
            new_data = []
            for tok_str in data:
                i, new_tok_str = 0, []
                while i < len(tok_str):
                    if i < len(tok_str) - 1 and (tok_str[i], tok_str[i+1]) == best_pair:
                        new_tok_str.append(new_id)
                        i += 2
                    else:
                        new_tok_str.append(tok_str[i])
                        i += 1
                new_data.append(new_tok_str)
            data = new_data

        # END CODE

    def pad(self, tokens, length):
        """ Pads a tokenized string to a specified length, for batch processing.

        Args:
            tokens (list[int]): Encoded token string to be padded.
            length (int): Length of tokens to pad to.

        Returns:
            list[int]: Token string padded to desired length.
        """

        # BEGIN CODE : tokenizer.pad

        # ADD YOUR CODE HERE
        if len(tokens) < length:
            tokens = [ *tokens ]
            tokens += ([ self.special_tokens['[PAD]'] ] * (length - len(tokens)))

        return tokens

        # END CODE

    def unpad(self, tokens):
        """ Removes padding from a token string.

        Args:
            tokens (list[int]): Encoded token string with padding.

        Returns:
            list[int]: Token string with padding removed.
        """

        # BEGIN CODE : tokenizer.unpad

        # ADD YOUR CODE HERE
        no_pad_len = len(tokens)
        while tokens[no_pad_len-1] == self.special_tokens['[PAD]']: no_pad_len -= 1

        return tokens[:no_pad_len]

        # END CODE

    def get_special_tokens(self):
        """ Returns the associated special tokens.

            Returns:
                dict[str, int]: Mapping describing the special tokens, if any.
                    This is a mapping between a string segment (token) and its associated id (token_id).
        """

        # BEGIN CODE : tokenizer.get_special_tokens

        # ADD YOUR CODE HERE
        return self.special_tokens

        # END CODE

    def get_vocabulary(self):
        """ Returns the learnt vocabulary post the training process.

            Returns:
                dict[str, int]: Mapping describing the vocabulary and special tokens, if any.
                    This is a mapping between a string segment (token) and its associated id (token_id).
        """

        # BEGIN CODE : tokenizer.get_vocabulary

        # ADD YOUR CODE HERE
        return self.vocab

        # END CODE

    def encode(self, string, add_start=True, add_end=True):
        """ Encodes a string into a list of tokens.

        Args:
            string (str): Input string to be tokenized.
            add_start (bool): If true, adds the start of sequence token.
            add_end (bool): If true, adds the end of sequence token.
        Returns:
            list[int]: List of tokens (unpadded).
        """

        # BEGIN CODE : tokenizer.encode

        # ADD YOUR CODE HERE
        string = unicodedata.normalize('NFKC', string)

        tokens = [ i+len(self.special_tokens) for i in string.encode('utf-8') ]

        while len(tokens) > 1:
            pairs = set()
            for tok, next_tok in zip(tokens, tokens[1:]):
                pairs.add((tok, next_tok))

            merge_pair = min(pairs, key=lambda x: self.merge_rules.get(x, float("inf")))
            if merge_pair not in self.merge_rules: break

            i, new_tokens = 0, []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge_pair:
                    new_tokens.append(self.merge_rules[merge_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        if add_start: tokens = [ self.special_tokens['[BOS]'] ] + tokens
        if add_end  : tokens = tokens + [ self.special_tokens['[EOS]'] ]

        return tokens

        # END CODE

    def decode(self, tokens, strip_special=True):
        """ Decodes a string from a list of tokens.
            Undoes the tokenization, returning back the input string.

        Args:
            tokens (list[int]): List of encoded tokens to be decoded. No padding is assumed.
            strip_special (bool): Whether to remove special tokens or not.

        Returns:
            str: Decoded string.
        """

        # BEGIN CODE : tokenizer.decode

        # ADD YOUR CODE HERE
        if strip_special:
            special_tokens = set(self.special_tokens.values())
            tokens = [ token for token in tokens if token not in special_tokens ]

        return (b''.join(self.inv_vocab[tok_id] for tok_id in tokens)).decode('utf-8', errors='replace')

        # END CODE

    def batch_encode(self, batch, padding=None, add_start=True, add_end=True):
        """Encodes multiple strings in a batch to list of tokens padded to a given size.

        Args:
            batch (list[str]): List of strings to be tokenized.
            padding (int, optional): Optional, desired tokenized length. Outputs will be padded to fit this length.
            add_start (bool): If true, adds the start of sequence token.
            add_end (bool): If true, adds the end of sequence token.

        Returns:
            list[list[int]]: List of tokenized outputs, padded to the same length.
        """

        batch_output = [ self.encode(string, add_start, add_end) for string in batch ]
        if padding:
            for i, tokens in enumerate(batch_output):
                if len(tokens) < padding:
                    batch_output[i] = self.pad(tokens, padding)
        return batch_output

    def batch_decode(self, batch, strip_special=True):
        """ Decodes a batch of encoded tokens to normal strings.

        Args:
            batch (list[list[int]]): List of encoded token strings, optionally padded.
            strip_special (bool): Whether to remove special tokens or not.

        Returns:
            list[str]: Decoded strings after padding is removed.
        """
        return [ self.decode(self.unpad(tokens), strip_special=strip_special) for tokens in batch ]

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

class RNNEncoderDecoderLM(torch.nn.Module):
    """ Implements an Encoder-Decoder network, using RNN units. """

    # Feel free to add additional parameters to __init__
    def __init__(self, src_vocab_size, tgt_vocab_size, embd_dims, hidden_size, num_layers=1, dropout=0.1):
        """ Initializes the encoder-decoder network, implemented via RNNs.

        Args:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            embd_dims (int): Embedding dimensions.
            hidden_size (int): Size/Dimensions for the hidden states.
        """

        super(RNNEncoderDecoderLM, self).__init__()

        # Dummy parameter to track the model device. Do not modify.
        self._dummy_param = torch.nn.Parameter(torch.Tensor(0), requires_grad=False)

        # BEGIN CODE : enc-dec-rnn.init

        # ADD YOUR CODE HERE
        self.enc_embd = torch.nn.Embedding(src_vocab_size, embd_dims)
        self.dec_embd = torch.nn.Embedding(tgt_vocab_size, embd_dims)

        self.enc_rnn = torch.nn.LSTM(
            num_layers=num_layers, hidden_size=hidden_size,
            input_size=embd_dims, dropout=dropout, batch_first=True
        )
        self.dec_rnn = torch.nn.LSTM(
            num_layers=num_layers, hidden_size=hidden_size,
            input_size=embd_dims, dropout=dropout, batch_first=True
        )

        self.dec_unproject = torch.nn.Linear(hidden_size, tgt_vocab_size)

        # END CODE

    @property
    def device(self):
        """ Returns the device the model parameters are on. """
        return self._dummy_param.device

    def forward(self, inputs, decoder_inputs, decoder_hidden_state=None):
        """ Performs a forward pass over the encoder-decoder network.

            Accepts inputs for the encoder, inputs for the decoder, and hidden state for
                the decoder to continue generation after the given input.

        Args:
            inputs (torch.Tensor): tensor of shape [batch_size?, max_seq_length]
            decoder_inputs (torch.Tensor): tensor of shape [batch_size?, 1]
            decoder_hidden_state (any): tensor to represent decoder hidden state from time step T-1.

        Returns:
            tuple[torch.Tensor, any]: output from the decoder, and associated hidden state for the next step.
            Decoder outputs should be log probabilities over the target vocabulary.
        """

        # BEGIN CODE : enc-dec-rnn.forward

        # ADD YOUR CODE HERE
        if decoder_hidden_state is None:
            lengths = torch.count_nonzero(inputs, dim=-1).cpu()
            embd_inputs = self.enc_embd(inputs)
            if len(lengths.shape) > 0:
                embd_inputs = torch.nn.utils.rnn.pack_padded_sequence(embd_inputs, lengths, True, enforce_sorted=False)
            _, decoder_hidden_state = self.enc_rnn(embd_inputs)

        dec_embd_inputs = self.dec_embd(decoder_inputs)
        dec_outputs, decoder_hidden_state = self.dec_rnn(dec_embd_inputs, decoder_hidden_state)
        dec_raw_logits = self.dec_unproject(dec_outputs)

        return torch.nn.functional.log_softmax(dec_raw_logits, dim=-1), decoder_hidden_state

        # END CODE

    def log_probability(self, seq_x, seq_y):
        """ Compute the conditional log probability of seq_y given seq_x, i.e., log P(seq_y | seq_x).

        Args:
            seq_x (torch.tensor): Input sequence of tokens.
            seq_y (torch.tensor): Output sequence of tokens.

        Returns:
            float: Log probability of seq_y given seq_x
        """

        # BEGIN CODE : enc-dec-rnn.log_probability

        # ADD YOUR CODE HERE
        self.eval()

        total_log_prob = 0

        with torch.no_grad():
            decoder_hidden_state = None
            for i in range(len(seq_y) - 1):
                logits, decoder_hidden_state = self(
                    seq_x, seq_y[i:i+1],
                    decoder_hidden_state
                )
                total_log_prob += logits[-1, seq_y[i+1]]

            return total_log_prob.item()

        # END CODE

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

    model.eval()

    with torch.no_grad():
        seq_x = torch.tensor(src_tokenizer.encode(seq_x)).to(model.device)
        seq_y = tgt_tokenizer.encode("आमिष", add_start=True, add_end=True)
        seq_y, eos_token = seq_y[:1], seq_y[-1]

        decoder_hidden_state = None

        for _ in range(max_length):
            logits, decoder_hidden_state = model(
                seq_x, torch.tensor(seq_y[-1:], device=model.device),
                decoder_hidden_state
            )
            seq_y.append(logits[-1].cpu().argmax().item())
            if seq_y[-1] == eos_token: break

        return tgt_tokenizer.decode(seq_y)
    pass

class RNNEncoderDecoderTrainer:
    """ Performs model training in a model-agnostic manner.
        Requires specifying the model instance, the loss criterion to optimize,
          the optimizer to use and the directory to save data to.
    """

    def __init__(self, model, criterion, optimizer):
        """ Initializes the trainer.

        Args:
            directory (str): Directory to save checkpoints and the model data in.
            model (torch.nn.Module): Torch model (must inherit `torch.nn.Module`) to train.
            criterion (torch.nn.Function): Loss criterion, i.e., the loss function to optimize for training.
            optimizer (torch.optim.Optimizer): Optimizer to use for training.
        """

        self.model            = model
        self.optimizer        = optimizer
        self.criterion        = criterion
        self.last_checkpoint  = 0
        self.loss_history     = { 'train': [], 'valid': [] }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def make_dataloader(dataset, shuffle_data=True, batch_size=8, collate_fn=None):
        """ Create a dataloader for a torch Dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to process.
            shuffle_data (bool, optional): If true, shuffles the data. Defaults to True.
            batch_size (int, optional): Number of items per batch. Defaults to 8.
            collate_fn (function, optional): Function to use for collating instances to a batch.

        Returns:
            torch.utils.data.DataLoader: Dataloader over the given data, post processing.
        """

        # BEGIN CODE : trainer.make_dataloader

        # ADD YOUR CODE HERE
        return DataLoader(dataset, shuffle=shuffle_data, batch_size=batch_size, collate_fn=collate_fn)

        # END CODE

    def train_step(self, x_batch, y_batch):
        """ Performs a step of training, on the training batch.

        Args:
            x_batch (torch.Tensor): Input batch.
            y_batch (torch.Tensor): Output batch.

        Returns:
            float: Training loss with the current model, on this batch.
        """

        # BEGIN CODE : trainer.train_step

        # ADD YOUR CODE HERE
        self.optimizer.zero_grad()

        y_batch_pred, decoder_hidden_state = [], None

        for i in range(y_batch.shape[1]-1):
            logits, decoder_hidden_state = self.model(
                x_batch.to(self.device),
                y_batch[:, i:i+1].to(self.device),
                decoder_hidden_state
            )
            y_batch_pred.append(logits)

        y_batch_pred = torch.cat(y_batch_pred, dim=1)

        y_batch_pred = y_batch_pred.reshape(-1, y_batch_pred.shape[-1])
        y_batch      = y_batch[:, 1:].reshape(-1).to(self.device)

        loss = self.criterion(y_batch_pred, y_batch)
        loss [y_batch == 0] = 0

        loss = loss.sum() / (y_batch != 0).sum()
        loss.backward()
        self.optimizer.step()

        return loss.item()

        # END CODE

    def eval_step(self, validation_dataloader):
        """ Perfoms an evaluation step, on the validation dataloader.

        Args:
            validation_dataloader (torch.utils.data.DataLoader): Dataloader for the validation dataset.

        Returns:
            float: Validation loss with the current model checkpoint.
        """

        # BEGIN CODE : trainer.eval_step

        # ADD YOUR CODE HERE
        self.model.eval()

        val_loss = 0

        with torch.no_grad():
            for ex_batch, ey_batch in validation_dataloader:
                ey_batch_pred, decoder_hidden_state = [], None

                for i in range(ey_batch.shape[1]-1):
                    logits, decoder_hidden_state = self.model(
                        ex_batch.to(self.device),
                        ey_batch[:, i:i+1].to(self.device),
                        decoder_hidden_state
                    )
                    ey_batch_pred.append(logits)

                ey_batch_pred = torch.cat(ey_batch_pred, dim=1)
                ey_batch_pred = ey_batch_pred.view(-1, ey_batch_pred.shape[-1])
                ey_batch      = ey_batch[:, 1:].reshape(-1).to(self.device)

                loss = self.criterion(ey_batch_pred, ey_batch)
                loss [ey_batch == 0] = 0

                loss = loss.sum() / (ey_batch != 0).sum()

                val_loss += loss.item()

        self.model.train()

        return val_loss / len(validation_dataloader)

        # END CODE

    def train(self, train_dataset, validation_dataset=None,
              num_epochs=10, batch_size=8, shuffle=True,
              eval_steps=100, collate_fn=None):
        """ Handles the training loop for the model.

        Args:
            train_dataset (torch.utils.data.Dataset): Dataset to train on.
            validation_dataset (torch.utils.data.Dataset, optional): Data to validate on. Defaults to None.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 10.
            batch_size (int, optional): Number of items to process per batch. Defaults to 8.
            shuffle (bool, optional): Whether to shuffle the data or not. Defaults to True.
            eval_steps (int, optional): Number of steps post which the model should be evaluated. Defaults to 100.
            collate_fn (function, optional): Function to use for collating instances to a batch.
        """

        current_checkpoint = 0
        self.model.to(self.device)
        self.model.train()

        with tqdm.tqdm(total = math.ceil(len(train_dataset) / batch_size) * num_epochs) as pbar:
            for epoch in range(num_epochs):
                train_dataloader      = self.make_dataloader(train_dataset, shuffle, batch_size, collate_fn)
                if validation_dataset is not None:
                    validation_dataloader = self.make_dataloader(validation_dataset, shuffle, batch_size, collate_fn)

                for batch, (x_batch, y_batch) in enumerate(train_dataloader):
                    pbar.set_description(f"Epoch {epoch+1} / {num_epochs}")

                    # If we are resuming training, skip this iteration
                    if current_checkpoint < self.last_checkpoint:
                        current_checkpoint += 1
                        pbar.update()
                        continue

                    # Do a step of training
                    loss = self.train_step(x_batch, y_batch)
                    self.loss_history['train'].append(loss)
                    pbar.set_postfix({ 'batch': batch+1, 'loss': loss })

                    current_checkpoint += 1
                    pbar.update()

                    # Evaluate after every eval_steps
                    if (current_checkpoint) % eval_steps == 0:
                        if validation_dataset is not None:
                            val_loss = self.eval_step(validation_dataloader)
                            self.loss_history['valid'].append(val_loss)
                        else:
                            val_loss = None

                        print('[>]', f"epoch #{epoch+1:{len(str(num_epochs))}},",
                              f"batch #{batch+1:{len(str(len(train_dataloader)))}}:",
                              "loss:", f"{loss:.8f}", '|', "val_loss:", f"{val_loss:.8f}")

def get_tokenizer_performance(src_tokenizer, tgt_tokenizer, train_data, test_data):
    rnn_enc_dec_params = {
        'src_vocab_size': len(src_tokenizer.get_vocabulary()) + 1,
        'tgt_vocab_size': len(tgt_tokenizer.get_vocabulary()) + 1,
        'embd_dims'     : 256,
        'hidden_size'   : 512,
        'dropout'       : 0,
        'num_layers'    : 1
    }

    rnn_enc_dec_data_params = dict(
        src_padding=100,
        tgt_padding=100,
    )

    rnn_enc_dec_training_params = dict(
        num_epochs=3,
        batch_size=16,
        shuffle=True,
        eval_steps=40000
    )

    torch.manual_seed(42)

    model = RNNEncoderDecoderLM(**rnn_enc_dec_params)

    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.NLLLoss(reduction='none')
    trainer = RNNEncoderDecoderTrainer(model, criterion, optimizer)

    train_dataset      = TokenizerDataset(train_data, src_tokenizer, tgt_tokenizer, **rnn_enc_dec_data_params)
    validation_dataset = TokenizerDataset(test_data , src_tokenizer, tgt_tokenizer, **rnn_enc_dec_data_params)

    rnn_enc_dec_train_data = dict(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        collate_fn=train_dataset.collate
    )

    trainer.train(**rnn_enc_dec_train_data, **rnn_enc_dec_training_params)

    evaluator = Evaluator(src_tokenizer, tgt_tokenizer)
    evaluator.set_decoding_method(rnn_greedy_generate)

    return evaluator.evaluate(
        model, test_data['Name'], test_data['Translation'],
        max_length=rnn_enc_dec_data_params['tgt_padding']
    )

# ==== Tests

class Context:
    def __init__(self, module, record, data_dir) -> None:
        self.module = module
        self.record = record
        self.data_dir = data_dir
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.SRC_VOCAB_SIZE = eval(record.deepget(("variables", "SRC_VOCAB_SIZE"), "900"))

        self.train_data      = read_dataframe("train", os.path.join(data_dir, "data"))
        self.validation_data = read_dataframe("valid", os.path.join(data_dir, "data"))
        self.test_data       = read_dataframe("test", os.path.join(data_dir, "data"))

    def load_tokenizers(self, train=False):
        init_tokenizers = getattr(self.module, 'init_tokenizers')

        if train:
            src, tgt = init_tokenizers(True, train_data=self.train_data, validation_data=self.validation_data)
        else:
            root     = self.record.deepget("meta.root")
            src_file = self.record.deepget(("files", "found", "src-tokenizer/tokenizer.pkl"))
            tgt_file = self.record.deepget(("files", "found", "tgt-tokenizer/tokenizer.pkl"))

            if root is None or src_file is None or tgt_file is None:
                raise ValueError("Tokenizer file(s) not found")

            src, tgt = init_tokenizers(
                False,
                train_data=self.train_data, validation_data=self.validation_data,
                src_file=os.path.join(root, "src-tokenizer"),
                tgt_file=os.path.join(root, "tgt-tokenizer")
            )

        self.src_tokenizer, self.tgt_tokenizer = src, tgt

def test_tokenizer_creation(context: Context):
    context.load_tokenizers(train=True)
    assert context.src_tokenizer is not None
    assert context.tgt_tokenizer is not None

def test_tokenizer_functions(context: Context):
    if context.src_tokenizer is None:
        context.load_tokenizers()

    names = [ "kirisame", "parul", "alaina", "aamish" ]

    encoded = [ context.src_tokenizer.encode(name) for name in names ]
    decoded_1 = [ context.src_tokenizer.decode(name) for name in encoded ]
    decoded_2 = [ context.src_tokenizer.decode(name) for name in encoded ]

    assert all(dec1 == dec2 for dec1, dec2 in zip(decoded_1, decoded_2))

    assert len(context.src_tokenizer.pad(encoded[0], 2*len(encoded[0]))) == 2*len(encoded[0])

    assert len(context.src_tokenizer.unpad(
        context.src_tokenizer.pad(encoded[0], 2*len(encoded[0]))
    )) == len(encoded[0])

    spl_tokens = context.src_tokenizer.get_special_tokens()
    assert isinstance(spl_tokens, dict)

    vocabulary = context.src_tokenizer.get_vocabulary()
    assert isinstance(vocabulary, dict)

    assert len(vocabulary) <= len(spl_tokens) + context.SRC_VOCAB_SIZE

    assert len(context.src_tokenizer.decode(encoded[0], strip_special=True)) != \
        len(context.src_tokenizer.decode(encoded[0], strip_special=False))

def test_tokenizer_consistency(context: Context):
    if context.src_tokenizer is None:
        context.load_tokenizers()

    old_src_tokenizer = context.src_tokenizer
    old_tgt_tokenizer = context.tgt_tokenizer

    context.load_tokenizers(train=True)

    old_src_vars = {
        name: var for name, var in vars(old_src_tokenizer).items()
        if not callable(var)
    }
    new_src_vars = {
        name: var for name, var in vars(context.src_tokenizer).items()
        if not callable(var)
    }
    old_tgt_vars = {
        name: var for name, var in vars(old_tgt_tokenizer).items()
        if not callable(var)
    }
    new_tgt_vars = {
        name: var for name, var in vars(context.tgt_tokenizer).items()
        if not callable(var)
    }

    for name, var in old_src_vars.items():
        assert name in new_src_vars
        assert var == new_src_vars[name], "assert fail (src): " + name

    for name, var in old_tgt_vars.items():
        assert name in new_tgt_vars
        assert var == new_tgt_vars[name], "assert fail (tgt): " + name

def test_tokenizer_behavior(context: Context):
    if context.src_tokenizer is None:
        context.load_tokenizers()

    instances = context.test_data['Translation'].sample(n=5, random_state=20240401).tolist()

    max_length = max(map(
        lambda x: len(context.tgt_tokenizer.encode(x, add_start=False, add_end=False)),
        instances
    ))

    # Batch encode all instances with 'max' padding':
    tokenized_instances = context.tgt_tokenizer.batch_encode(
        instances, padding=max_length, add_start=False, add_end=False
    )

    # Check if length of encoded strings is consistent with the expected length.
    assert all(len(tok_str) == max_length for tok_str in tokenized_instances)

    max_length = max(map(
        lambda x: len(context.tgt_tokenizer.encode(x, add_start=True, add_end=True)),
        instances
    ))

    # Batch encode all instances with 'max' padding':
    tokenized_instances = context.tgt_tokenizer.batch_encode(
        instances, padding=max_length, add_start=True, add_end=True
    )

    # Check if length of encoded strings is consistent with the expected length.
    assert all(len(tok_str) == max_length for tok_str in tokenized_instances)

    # # Check if all strings start with the correct 'start' tokens.
    assert all(tok_str[0] == tokenized_instances[0][0] for tok_str in tokenized_instances)

    # # Check if all strings end with the correct 'end' tokens.
    end_i = [ i for i, seq in enumerate(tokenized_instances) if len(context.tgt_tokenizer.unpad(seq)) == max_length ]
    pad_i = [ i for i, seq in enumerate(tokenized_instances) if len(context.tgt_tokenizer.unpad(seq)) <  max_length ]

    assert all(
        tokenized_instances[i][-1] == tokenized_instances[end_i[0]][-1]
        for i in end_i
    )
    assert all(
        tokenized_instances[i][-1] == tokenized_instances[pad_i[0]][-1]
        for i in pad_i
    )
    pad_lengths = [ tokenized_instances[i].index(tokenized_instances[end_i[0]][-1]) for i in pad_i ]
    assert all(
        all(tok == tokenized_instances[pad_i[0]][-1] for tok in tokenized_instances[i][plen+1:])
        for i, plen in zip(pad_i, pad_lengths)
    )

def test_tokenizer_quality(context: Context):
    if context.src_tokenizer is None:
        context.load_tokenizers()

    vocabulary = context.src_tokenizer.get_vocabulary()
    spl_tokens = context.src_tokenizer.get_special_tokens()
    if context.SRC_VOCAB_SIZE is not None:
        assert len(vocabulary) <= len(spl_tokens) + context.SRC_VOCAB_SIZE

    tok_length = len(context.tgt_tokenizer.encode("श्रीकृष्णकुमारचंदन", add_start=False, add_end=False))
    assert tok_length < len("श्रीकृष्णकुमारचंदन") + 3, "tokenizer produces too many tokens"

    tok_impl_scores = context.record.deepget("outputs.tokenizer.scores")

    if tok_impl_scores is None:
        tok_impl_scores = get_tokenizer_performance(
            context.src_tokenizer, context.tgt_tokenizer,
            context.train_data, context.test_data.sample(n=256, random_state=20240401, ignore_index=True)
        )

    # trained_src_tokenizer = BPETokenizer()
    # trained_src_tokenizer.train(context.train_data['Name'], 1200)

    # trained_tgt_tokenizer = BPETokenizer()
    # trained_tgt_tokenizer.train(context.train_data['Translation'], 1200)

    # ref_impl_scores = get_tokenizer_performance(
    #     trained_src_tokenizer, trained_tgt_tokenizer,
    #     context.train_data, context.test_data.sample(n=256, random_state=20240401, ignore_index=True)
    # )

    ref_impl_scores = {
        "accuracy": 0.20,
        "cer"     : 0.25,
        "ter"     : 0.40,
        "bleu"    : 0.55
    }

    try:
        assert ref_impl_scores['bleu'] <= tok_impl_scores['bleu'], \
            "assertion fail: tokenizer no good than overinitialized BPE!"
        assert ref_impl_scores['accuracy'] <= tok_impl_scores['accuracy'], \
            "assertion fail: tokenizer no good than overinitialized BPE!"
        assert ref_impl_scores['cer'] >= tok_impl_scores['cer'], \
            "assertion fail: tokenizer no good than overinitialized BPE!"
        assert ref_impl_scores['ter'] >= tok_impl_scores['ter'], \
            "assertion fail: tokenizer no good than overinitialized BPE!"

        return { 'scores': tok_impl_scores }
    except AssertionError as exc:
        setattr(exc, 'outputs', { 'scores': tok_impl_scores })
        raise exc