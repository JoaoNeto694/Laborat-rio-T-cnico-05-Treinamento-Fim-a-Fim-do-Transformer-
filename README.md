# Laboratório Técnico 05: Treinamento Fim-a-Fim do Transformer

Treinamento supervisionado de um Transformer completo (Encoder-Decoder) para tradução EN→DE usando o dataset Multi30k, tokenização multilingual e loop de otimização com Adam.

---

## Pré-requisitos

- Python 3.8+
- PyTorch
- Hugging Face `datasets`
- Hugging Face `transformers`

Instale as dependências com:

```bash
pip install torch datasets transformers
```

---

## Como rodar

```bash
python transformer_treinamento.py
```

---

## O que o código faz

### Componentes implementados

| Componente | Descrição |
|---|---|
| `MultiHeadAttention` | Multi-head attention com pesos locais por cabeça e projeção global — mesma arquitetura Option B dos labs anteriores |
| `CrossAttention` | Atenção cruzada: Q vem do decoder, K e V vêm da saída do encoder |
| `FeedForwardNetwork` | Duas camadas lineares com ativação ReLU |
| `EncoderBlock` | Self-attention + FFN com Add & Norm |
| `DecoderBlock` | Masked self-attention + cross-attention + FFN com Add & Norm |
| `Transformer` | Pilha completa com embeddings de src/tgt, N blocos de encoder e decoder, e projeção final |

### Fluxo de execução

**1. Preparação do dataset (Tarefa 1)**
São carregadas 1000 amostras do dataset `bentrevett/multi30k` (pares EN→DE). As frases em inglês são usadas como entrada do encoder, as frases em alemão como alvo do decoder.

**2. Tokenização (Tarefa 2)**
O tokenizador `bert-base-multilingual-cased` processa ambas as línguas. Cada sequência-alvo recebe `<START>` no início e `<EOS>` no final. As sequências são truncadas e padded para comprimento fixo.

**3. Loop de treinamento (Tarefa 3)**
A cada epoch e batch:
- `src_b` passa pelo encoder `encoder_out`
- `tgt_input` (sequência-alvo sem o último token) passa pelo decoder junto com `encoder_out`
- Os logits são comparados com `tgt_target` (sequência-alvo sem o `<START>`) via `CrossEntropyLoss`
- `loss.backward()` + `optimizer.step()` atualizam todos os pesos

**4. Inferência auto-regressiva (Tarefa 4)**
A partir de `<START>`, o decoder gera tokens um a um (greedy decoding). A cada passo o token com maior logit é escolhido e anexado à sequência até gerar `<EOS>` ou atingir o limite de 30 passos.

---

## Configuração padrão

```python
d_model = 128 # Dimensão dos embeddings
h = 4 # Número de cabeças de atenção
N = 2 # Número de blocos empilhados
d_ffn = 256 # Dimensão interna do FFN
MAX_LEN = 30 # Comprimento máximo das sequências
BATCH_SIZE = 16 # Tamanho do batch
N_EPOCHS = 40 # Número de epochs (reduzir para testes rápidos)
```

---

## Saída esperada

A cada epoch o modelo imprime a loss média do conjunto de treinamento:

```
 1/40  |  Loss: 7.1832
 2/40  |  Loss: 6.4201
...
40/40  |  Loss: 1.3047
```

Ao final, o modelo tenta reproduzir a primeira frase do dataset (teste de overfitting):

```
Frase de entrada (EN): Two young, White males are outside near many bushes.
  Passo  1: 'Zwei' (id=15421)
  Passo  2: 'junge' (id=11724)
  ...
Tradução esperada : Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
Tradução gerada   : Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
```

> Com N_EPOCHS = 40 o modelo tende a memorizar bem as frases do conjunto de treino. Para execução mais rápida, dá para reduzir para o número que o lab recomenda, mas é bem menos preciso.

---

## Observações

- A arquitetura Multi-Head Attention segue a **Option B** dos laboratórios anteriores: cada cabeça projeta Q, K e V individualmente; os resultados são concatenados e repassados por matrizes globais W_Q_G, W_K_G e W_V_G antes da operação de atenção unificada.
- Os pesos são inicializados com `* 0.1` para evitar colapso do softmax.
- O padding é ignorado na loss via `ignore_index=PAD_TOKEN_ID`.
- O otimizador Adam é o mesmo utilizado no paper original *"Attention Is All You Need"*.

---

## Uso de IA generativa

- Reescrita do código base do Lab 04 em PyTorch (loss muito alto com NumPy puro).
- Sintaxe do python e torch
- Geração deste README.
