import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# CÓDIGO DO LAB 04 (base para este laboratório)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)
    return weights @ V, weights

def layer_norm(X, epsilon=1e-6):
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X,  axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + epsilon)

def create_causal_mask(seq_len):
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

class MultiHeadAttention:
    def __init__(self, d_model, h):
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model
        self.W_Q = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]
        self.W_K = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]
        self.W_V = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(h)]
        self.W_Q_G = np.random.randn(d_model, d_model) * 0.1
        self.W_K_G = np.random.randn(d_model, d_model) * 0.1
        self.W_V_G = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X, mask=None):
        Qs = [X @ wq for wq in self.W_Q]
        Ks = [X @ wk for wk in self.W_K]
        Vs = [X @ wv for wv in self.W_V]
        Q = np.concatenate(Qs, axis=-1) @ self.W_Q_G
        K = np.concatenate(Ks, axis=-1) @ self.W_K_G
        V = np.concatenate(Vs, axis=-1) @ self.W_V_G
        output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        return output

class FeedForwardNetwork:
    def __init__(self, d_model, d_ffn):
        self.W1 = np.random.randn(d_model, d_ffn) * 0.1
        self.b1 = np.zeros(d_ffn)
        self.W2 = np.random.randn(d_ffn, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        return np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2

class EncoderBlock:
    def __init__(self, d_model, h, d_ffn):
        self.mha = MultiHeadAttention(d_model, h)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, X):
        X = layer_norm(X + self.mha.forward(X))
        X = layer_norm(X + self.ffn.forward(X))
        return X

class DecoderBlock:
    def __init__(self, d_model, h, d_ffn):
        self.mha = MultiHeadAttention(d_model, h)
        self.mha_cross = MultiHeadAttention(d_model, h)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, X, encoder_out):
        mask = create_causal_mask(X.shape[-2])
        X = layer_norm(X + self.mha.forward(X, mask))
        X = layer_norm(X + self.mha_cross.forward(X))
        X = layer_norm(X + self.ffn.forward(X))
        return X

# TAREFA 1 - Preparando o Dataset Real (Hugging Face)
dataset = load_dataset("bentrevett/multi30k", split="train")
dataset = dataset.select(range(1000))

src_sentences = [ex["en"] for ex in dataset]
tgt_sentences = [ex["de"] for ex in dataset]

# TAREFA 2 - Tokenização Básica
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

START_TOKEN_ID = tokenizer.cls_token_id 
EOS_TOKEN_ID   = tokenizer.sep_token_id
PAD_TOKEN_ID   = tokenizer.pad_token_id  

MAX_LEN = 30

def tokenize_pair(src, tgt):
    # Converte um par de frases em listas de inteiros com tokens especiais.
    src_ids = tokenizer.encode(src, add_special_tokens=False)[:MAX_LEN]
    tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)[:MAX_LEN - 2]
    # Adiciona <START> e <EOS> na frase de destino (Decoder)
    tgt_ids = [START_TOKEN_ID] + tgt_ids + [EOS_TOKEN_ID]
    return src_ids, tgt_ids

def pad_sequence(seq, max_len, pad_id):
    return seq + [pad_id] * (max_len - len(seq))

# Tokeniza todas as 1000 frases
src_encoded = []
tgt_encoded = []
for src, tgt in zip(src_sentences, tgt_sentences):
    s, t = tokenize_pair(src, tgt)
    src_encoded.append(pad_sequence(s, MAX_LEN, PAD_TOKEN_ID))
    tgt_encoded.append(pad_sequence(t, MAX_LEN + 1, PAD_TOKEN_ID))

src_encoded = np.array(src_encoded, dtype=np.int32)
tgt_encoded = np.array(tgt_encoded, dtype=np.int32)

# Decoder input: tudo menos o último token
# Decoder target: tudo menos o primeiro token (<START>)
tgt_input  = tgt_encoded[:, :-1] 
tgt_target = tgt_encoded[:, 1:] 