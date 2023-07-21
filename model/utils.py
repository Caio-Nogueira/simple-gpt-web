import torch
import torch.nn.functional as F
from torchtext import datasets
from torchtext.data import get_tokenizer
import torch.nn as nn
from model import LanguageModel


tokenizer = get_tokenizer("basic_english")
data_path = "/home/caio/gpt/data/wikitext-2/wiki.train.tokens"

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("\n", "<eos>")

tokens = tokenizer(text)
distinct_tokens = sorted(list(set(tokens)))

token_to_idx = {token: idx for (idx, token) in enumerate(distinct_tokens)}
idx_to_token = {idx: token for (idx, token) in enumerate(distinct_tokens)}

N = int(0.9*len(tokens))
train_data = torch.as_tensor([token_to_idx[x] for x in tokens[:N]])
val_data = torch.as_tensor([token_to_idx[x] for x in tokens[N:]])

# HYPERPARAMETERS
n_embed = 512
n_head = 16
n_blocks = 6
batch_size = 16
ffdim = 2048
lr = 1e-3
eval_iters = 100
eval_interval = 500
max_iters = 2000
context_len = 32
vocab_size = len(token_to_idx.keys())
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(data):
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in ix])
    y = torch.stack([data[i+1:i+context_len+1] for i in ix])
    return x.to(device), y.to(device)

def build_attention_mask(batch_size):
    T=n_embed
    tril = torch.tril(torch.ones(n_embed, n_embed))
    attention_mask = torch.zeros((T,T))
    attention_mask = attention_mask.masked_fill(tril == 0, float('-inf'))
    attention_mask = F.softmax(attention_mask, dim=-1)

    attention_mask = torch.stack([attention_mask for i in range(batch_size)])
    return attention_mask.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for data, split in [(train_data, "train"), (val_data, "val")]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data)

            logits = model(X, build_attention_mask(X.size(0)))
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = Y.view(-1) # flatten targets list

            loss = F.cross_entropy(logits, targets)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(model):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data)

        logits = model(xb, build_attention_mask(xb.size(0)))
        B,T,C = logits.shape

        # Merge Time dimension
        logits = logits.view(B*T, C)
        targets = yb.view(-1) # flatten targets list

        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


model = LanguageModel(vocab_size, n_embed, n_head, ffdim, n_blocks, context_len)
model.to(device)
train(model)