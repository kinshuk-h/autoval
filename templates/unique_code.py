class Tokenizer:
    def __init__(self):
        # >>> {segment:tokenizer.init} <<<
        pass

    def train(self, data, vocab_size):
        # >>> {segment:tokenizer.train} <<<
        pass

    def pad(self, tokens, length):
        # >>> {segment:tokenizer.pad} <<<
        pass

    def unpad(self, tokens):
        # >>> {segment:tokenizer.unpad} <<<
        pass

    def get_special_tokens(self):
        # >>> {segment:tokenizer.get_special_tokens} <<<
        pass

    def get_vocabulary(self):
        # >>> {segment:tokenizer.get_vocabulary} <<<
        pass

    def encode(self, string, add_start=True, add_end=True):
        # >>> {segment:tokenizer.encode} <<<
        pass

    def decode(self, tokens, strip_special=True):
        # >>> {segment:tokenizer.decode} <<<
        pass

class RNNEncoderDecoderLM:
    def __init__(self, src_vocab_size, tgt_vocab_size, embd_dims, hidden_size, num_layers=1, dropout=0.1):
        # >>> {segment:enc-dec-rnn.init} <<<
        pass

    def forward(self, inputs, decoder_inputs, decoder_hidden_state=None):
        # >>> {segment:enc-dec-rnn.forward} <<<
        pass

    def log_probability(self, seq_x, seq_y):
        # >>> {segment:enc-dec-rnn.log_probability} <<<
        pass

class RNNEncoderDecoderTrainer:
    @staticmethod
    def make_dataloader(dataset, shuffle_data=True, batch_size=8, collate_fn=None):
        # >>> {segment:rnn-enc-dec-trainer.make_dataloader} <<<
        pass

    def train_step(self, x_batch, y_batch):
        # >>> {segment:rnn-enc-dec-trainer.train_step} <<<
        pass

    def eval_step(self, validation_dataloader):
        # >>> {segment:rnn-enc-dec-trainer.eval_step} <<<
        pass

class AttentionModule:
    def __init__(self, input_size):
        # >>> {segment:attn.init} <<<
        pass

    def forward(self, encoder_outputs, decoder_hidden_state, attn_mask=None):
        # >>> {segment:attn.forward} <<<
        pass

class RNNEncoderDecoderLMWithAttention:
    def __init__(self,src_vocab_size, tgt_vocab_size, embd_dims, hidden_size, num_layers=1, dropout=0.1):
        # >>> {segment:enc-dec-rnn-attn.init} <<<
        pass

    def log_probability(self, seq_x, seq_y):
        # >>> {segment:enc-dec-rnn-attn.probability} <<<
        pass

    def attentions(self, seq_x, terminate_token, max_length):
        # >>> {segment:enc-dec-rnn-attn.attentions} <<<
        pass

    def forward(self, inputs, decoder_inputs=None, decoder_hidden_state=None, output_attention=False):
        # >>> {segment:enc-dec-rnn-attn.forward} <<<
        pass

def rnn_greedy_generate(model, seq_x, src_tokenizer, tgt_tokenizer, max_length):
    # >>> {segment:enc-dec-rnn.greedy_generate} <<<
    pass