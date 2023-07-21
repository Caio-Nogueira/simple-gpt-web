import torch
from torch import nn
from torchtext.data import get_tokenizer
from torch.nn import functional as F



class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embed, n_head, ffdim, n_blocks, context_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.ffdim = ffdim
        self.n_blocks = n_blocks
        self.context_len = context_len



        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.positional_embeddings= nn.Embedding(context_len, n_embed)

        self.decoder = nn.TransformerDecoderLayer(d_model=n_embed, nhead=n_head, dim_feedforward=ffdim, batch_first=True)
        self.transformer = nn.TransformerDecoder(self.decoder, num_layers=n_blocks)

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, attention_mask): # x: (B, T, V)
        token_emb = self.token_embeddings(x) # (B, T, V) -> (B, T, C)
        pos_emb = self.positional_embeddings(torch.arange(self.context_len, device=x.device)) # (T, C)

        x = token_emb + pos_emb # (B, T, C)

        x = self.transformer(x, attention_mask)
        x = self.lm_head(x)
        return x

    def generate(self, idx, new_tokens, attention_mask):
        for token in range(new_tokens):

            batch_dim = idx.size(0)
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.context_len:]
            logits = self(idx_cond, attention_mask)

            # focus only on the last time step (last character)
            logits = logits[:, -1, :]
            B,C = logits.shape

            probs = F.softmax(logits, dim=-1) #(B ,C)
            new_idx = torch.multinomial(probs, num_samples=1) # (B,1) --> randomly selects samples based on a prob distribution given by softmax

            new_idx = new_idx.view(B, 1)
            idx = torch.cat((idx, new_idx), dim=1) #(B,T+1)
        return idx