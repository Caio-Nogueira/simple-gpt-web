import torch
from torchtext.data import get_tokenizer
from model import LanguageModel


with open('wikitext-2/wiki.train.tokens', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

distinct_tokens = sorted(list(set(tokens)))

token_to_idx = {token: idx for (idx, token) in enumerate(distinct_tokens)}
idx_to_token = {idx: token for (idx, token) in enumerate(distinct_tokens)}


# Hyperparameters
n_embed = 128
n_head = 16
n_blocks = 4
batch_size = 16
lr = 1e-3
context_len = 32
vocab_size = len(token_to_idx.keys())
device = "cuda" if torch.cuda.is_available() else "cpu"

# Train
model = LanguageModel(vocab_size, n_embed=n_embed, n_head=n_head, context_len=context_len, device=device)
model.to(device)

x = torch.randn((batch_size, context_len), dtype=torch.long, device=device)

out = model(x)

print(out.shape)