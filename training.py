import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer

# CÓDIGO DO LAB 04 (base para este laboratório) — reescrito em
# PyTorch para que loss.backward() atualize TODOS os pesos
# (W_Q, W_K, W_V, FFN, embeddings, etc.)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.h   = h
        self.d_k = d_model // h
        # Pesos por cabeça
        self.W_Q = nn.ParameterList([nn.Parameter(torch.randn(d_model, self.d_k) * 0.1) for _ in range(h)])
        self.W_K = nn.ParameterList([nn.Parameter(torch.randn(d_model, self.d_k) * 0.1) for _ in range(h)])
        self.W_V = nn.ParameterList([nn.Parameter(torch.randn(d_model, self.d_k) * 0.1) for _ in range(h)])
        # Pesos globais que recebem a concatenação de todas as cabeças
        self.W_Q_G = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.W_K_G = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.W_V_G = nn.Parameter(torch.randn(d_model, d_model) * 0.1)

    def forward(self, X, mask=None):
        Qs = [X @ wq for wq in self.W_Q]
        Ks = [X @ wk for wk in self.W_K]
        Vs = [X @ wv for wv in self.W_V]
        Q = torch.cat(Qs, dim=-1) @ self.W_Q_G
        K = torch.cat(Ks, dim=-1) @ self.W_K_G
        V = torch.cat(Vs, dim=-1) @ self.W_V_G
        scores = Q @ K.transpose(-1, -2) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        return weights @ V


class CrossAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.h   = h
        self.d_k = d_model // h
        self.W_Q = nn.ParameterList([nn.Parameter(torch.randn(d_model, self.d_k) * 0.1) for _ in range(h)])
        self.W_K = nn.ParameterList([nn.Parameter(torch.randn(d_model, self.d_k) * 0.1) for _ in range(h)])
        self.W_V = nn.ParameterList([nn.Parameter(torch.randn(d_model, self.d_k) * 0.1) for _ in range(h)])
        self.W_Q_G = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.W_K_G = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.W_V_G = nn.Parameter(torch.randn(d_model, d_model) * 0.1)

    def forward(self, X, encoder_out):
        # Q vem do decoder, K e V vêm do encoder
        Qs = [X           @ wq for wq in self.W_Q]
        Ks = [encoder_out @ wk for wk in self.W_K]
        Vs = [encoder_out @ wv for wv in self.W_V]
        Q = torch.cat(Qs, dim=-1) @ self.W_Q_G
        K = torch.cat(Ks, dim=-1) @ self.W_K_G
        V = torch.cat(Vs, dim=-1) @ self.W_V_G
        scores  = Q @ K.transpose(-1, -2) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return weights @ V


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d_model, d_ffn) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(d_ffn))
        self.W2 = nn.Parameter(torch.randn(d_ffn, d_model) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, X):
        return torch.relu(X @ self.W1 + self.b1) @ self.W2 + self.b2


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ffn):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, X):
        X = torch.layer_norm(X + self.mha(X), [X.shape[-1]])
        X = torch.layer_norm(X + self.ffn(X), [X.shape[-1]])
        return X


class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ffn):
        super().__init__()
        self.mha       = MultiHeadAttention(d_model, h)
        self.mha_cross = CrossAttention(d_model, h)
        self.ffn       = FeedForwardNetwork(d_model, d_ffn)

    def forward(self, X, encoder_out):
        seq_len = X.shape[-2]
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        X = torch.layer_norm(X + self.mha(X, mask),              [X.shape[-1]])
        X = torch.layer_norm(X + self.mha_cross(X, encoder_out), [X.shape[-1]])
        X = torch.layer_norm(X + self.ffn(X),                    [X.shape[-1]])
        return X


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, h, N, d_ffn, pad_id):
        super().__init__()
        self.src_emb        = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb        = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, h, d_ffn) for _ in range(N)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, h, d_ffn) for _ in range(N)])
        self.linear_out     = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src_ids):
        Z = self.src_emb(src_ids)
        for block in self.encoder_blocks:
            Z = block(Z)
        return Z

    def decode(self, tgt_ids, encoder_out):
        Y = self.tgt_emb(tgt_ids)
        for block in self.decoder_blocks:
            Y = block(Y, encoder_out)
        return self.linear_out(Y)

    def forward(self, src_ids, tgt_ids):
        encoder_out = self.encode(src_ids)
        return self.decode(tgt_ids, encoder_out)


# TAREFA 1 - Preparando o Dataset Real (Hugging Face)
dataset = load_dataset("bentrevett/multi30k", split="train")
dataset = dataset.select(range(1000))

src_sentences = [ex["en"] for ex in dataset]
tgt_sentences = [ex["de"] for ex in dataset]

# TAREFA 2 - Tokenização Básica
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

START_TOKEN_ID = tokenizer.cls_token_id 
EOS_TOKEN_ID = tokenizer.sep_token_id  
PAD_TOKEN_ID = tokenizer.pad_token_id 

MAX_LEN = 30

def tokenize_pair(src, tgt):
    src_ids = tokenizer.encode(src, add_special_tokens=False)[:MAX_LEN]
    tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)[:MAX_LEN - 2]
    # Adiciona <START> e <EOS> na frase de destino (Decoder)
    tgt_ids = [START_TOKEN_ID] + tgt_ids + [EOS_TOKEN_ID]
    return src_ids, tgt_ids

def pad_sequence(seq, max_len, pad_id):
    return seq + [pad_id] * (max_len - len(seq))

src_encoded = []
tgt_encoded = []

for src, tgt in zip(src_sentences, tgt_sentences):
    s, t = tokenize_pair(src, tgt)
    src_encoded.append(pad_sequence(s, MAX_LEN, PAD_TOKEN_ID))
    tgt_encoded.append(pad_sequence(t, MAX_LEN + 1, PAD_TOKEN_ID))

src_encoded = torch.tensor(src_encoded, dtype=torch.long)  
tgt_encoded = torch.tensor(tgt_encoded, dtype=torch.long) 

# Decoder input: tudo menos o último token
# Decoder target: tudo menos o primeiro token (<START>)
tgt_input  = tgt_encoded[:, :-1]   
tgt_target = tgt_encoded[:, 1:]    


# TAREFA 3 - O Motor de Otimização (Training Loop)
# Instancia o modelo com dimensões viáveis para o laboratório
d_model = 128
h = 4
N = 2
d_ffn = 256

model = Transformer(
    src_vocab_size = tokenizer.vocab_size,
    tgt_vocab_size = tokenizer.vocab_size,
    d_model = d_model,
    h = h,
    N = N,
    d_ffn = d_ffn,
    pad_id = PAD_TOKEN_ID,
)

# Função de Perda: CrossEntropyLoss com ignore_index para não penalizar o padding
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

# Otimizador Adam (mesmo usado no paper original)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

BATCH_SIZE = 16
# Resultado muito satisfatório com 40, porém demora muito na minha máquina
N_EPOCHS = 40
N_SAMPLES = src_encoded.shape[0]

for epoch in range(1, N_EPOCHS + 1):
    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, N_SAMPLES, BATCH_SIZE):
        src_b  = src_encoded[start:start + BATCH_SIZE]
        tin_b  = tgt_input  [start:start + BATCH_SIZE]
        tout_b = tgt_target [start:start + BATCH_SIZE]

        # Passe o lote pelo Encoder
        # Passe o lote da língua destino (deslocado 1 posição) pelo Decoder
        logits = model(src_b, tin_b) 

        # Calcula o erro (Loss)
        B, T, V = logits.shape
        loss = criterion(
            logits.reshape(B * T, V),
            tout_b.reshape(B * T)
        )

        # Aplica loss.backward() e optimizer.step() para atualizar os pesos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    print(f"{epoch:>2d}/{N_EPOCHS}  |  Loss: {epoch_loss / n_batches:.4f}")

# TAREFA 4 - A Prova de Fogo (Overfitting Test)
# Nessa fase, minha frase gerada estava completamente errada, então eu perguntei ao Claude o que havia de errado.
# Ele me recomendou jogar tudo para pytorch, então pedi a ele que reescrevesse o código com a mesma lógica, mas usando PyTorch
# ao invés de numpy. O loss ficou bem mais baixo e o resultado ficou bem melhor
# Pega uma frase específica que estava no conjunto de treino
frase_teste_src = src_sentences[0]
frase_teste_tgt = tgt_sentences[0]
print(f"Frase de entrada (EN): {frase_teste_src}")

# Prepara o input do encoder
src_ids = tokenizer.encode(frase_teste_src, add_special_tokens=False)[:MAX_LEN]
src_ids = pad_sequence(src_ids, MAX_LEN, PAD_TOKEN_ID)
src_ids = torch.tensor([src_ids], dtype=torch.long)

model.eval()
with torch.no_grad():
    encoder_out = model.encode(src_ids)

# Loop auto-regressivo (igual ao Lab 04)
generated_ids = [START_TOKEN_ID]
MAX_STEPS = 30
with torch.no_grad():
    for step in range(MAX_STEPS):
        tgt_ids = torch.tensor([generated_ids], dtype=torch.long)
        logits  = model.decode(tgt_ids, encoder_out)

        # Pega o próximo token (greedy: maior probabilidade)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        generated_ids.append(next_id)

        token_str = tokenizer.decode([next_id])
        print(f"  Passo {step + 1:>2d}: '{token_str}' (id={next_id})")

        if next_id == EOS_TOKEN_ID:
            break

# Decodifica a sequência gerada
ids_sem_especiais = [i for i in generated_ids if i not in (START_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID)]
traducao = tokenizer.decode(ids_sem_especiais, skip_special_tokens=True)

print(f"\nTradução esperada : {frase_teste_tgt}")
print(f"Tradução gerada   : {traducao}")
