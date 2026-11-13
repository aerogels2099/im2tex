"""Image-to-LaTeX encoder-decoder: pretrained CNN encoder (stride 16) + transformer decoder."""
import os

import timm
import torch
import torch.nn.functional as F
from torch import nn


class Im2Tex(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8,
                 encoder: str = "convnext_tiny", pretrained: bool = True,
                 max_tokens: int = 512, max_rows: int = 64, max_cols: int = 256):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           encoder=encoder, max_tokens=max_tokens, max_rows=max_rows, max_cols=max_cols)
        # Stride-16 feature map from the third stage; timm drops the unused last stage.
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True,
                                         out_indices=(2,), in_chans=1)
        self.proj = nn.Linear(self.encoder.feature_info.channels()[-1], d_model)
        self.row_emb = nn.Embedding(max_rows, d_model)
        self.col_emb = nn.Embedding(max_cols, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        # nn.Embedding's default N(0, 1) makes the tied output layer's first logits ~sqrt(d_model) too large
        # (initial loss ~40 instead of ~ln(vocab) = 8); small embeddings start training near uniform.
        for emb in (self.tok_emb, self.pos_emb, self.row_emb, self.col_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def grow_vocab(self, vocab_size: int, init_from: int):
        """Add tokens up to vocab_size, each starting as a copy of token init_from (a new start token from
        <bos>), so a trained model can learn them without starting over."""
        old = self.tok_emb.weight.data
        self.tok_emb = nn.Embedding(vocab_size, old.shape[1]).to(old.device, old.dtype)
        self.tok_emb.weight.data[: len(old)] = old
        self.tok_emb.weight.data[len(old):] = old[init_from]
        self.head = nn.Linear(old.shape[1], vocab_size, bias=False).to(old.device, old.dtype)
        self.head.weight = self.tok_emb.weight
        self.config["vocab_size"] = vocab_size

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H, W) with H, W multiples of 16 -> memory (B, H/16 * W/16, d_model)."""
        f = self.encoder(images)[-1]
        _, _, h, w = f.shape
        pos = self.row_emb.weight[:h, None] + self.col_emb.weight[None, :w]
        return self.proj(f.flatten(2).transpose(1, 2)) + pos.reshape(1, h * w, -1)

    def decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Logits (B, T, vocab) for decoder inputs tokens (B, T), each position seeing only earlier ones."""
        t = tokens.shape[1]
        x = self.tok_emb(tokens) + self.pos_emb.weight[:t]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=tokens.device, dtype=x.dtype)
        return self.head(self.norm(self.decoder(x, memory, tgt_mask=mask, tgt_is_causal=True)))

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits (B, T, vocab) for decoder inputs tokens (B, T)."""
        return self.decode(self.encode(images), tokens)

    def _heads(self, x):
        """(N, T, d_model) -> (N, heads, T, d_head)."""
        n, t, _ = x.shape
        return x.view(n, t, self.config["n_heads"], -1).transpose(1, 2)

    def _cross_kv(self, memory):
        """Each decoder layer's cross-attention keys and values, (B, heads, S, d_head): computed once per image."""
        d = self.config["d_model"]
        out = []
        for layer in self.decoder.layers:
            w, b = layer.multihead_attn.in_proj_weight, layer.multihead_attn.in_proj_bias
            out.append((self._heads(F.linear(memory, w[d:2 * d], b[d:2 * d])),
                        self._heads(F.linear(memory, w[2 * d:], b[2 * d:]))))
        return out

    def _step(self, tokens, t, cross, cache, cache_len):
        """Logits (N, vocab) after the decoder inputs at position t, tokens (N,), for N = B * beams rows
        (each image's beams adjacent). Same result as decode()[:, t] in eval mode, but appends this position's
        self-attention keys and values to cache instead of recomputing earlier positions."""
        n, d = tokens.shape[0], self.config["d_model"]
        x = self.tok_emb(tokens)[:, None] + self.pos_emb.weight[t]
        for layer, (ck, cv), kv in zip(self.decoder.layers, cross, cache):
            q, k, v = (self._heads(y) for y in F.linear(layer.norm1(x), layer.self_attn.in_proj_weight,
                                                         layer.self_attn.in_proj_bias).chunk(3, -1))
            if kv[0] is None:
                kv[:] = [k.new_empty(n, k.shape[1], cache_len, k.shape[3]) for _ in range(2)]
            kv[0][:, :, t:t + 1], kv[1][:, :, t:t + 1] = k, v
            a = F.scaled_dot_product_attention(q, kv[0][:, :, :t + 1], kv[1][:, :, :t + 1])
            x = x + layer.self_attn.out_proj(a.transpose(1, 2).reshape(n, 1, d))
            w, b = layer.multihead_attn.in_proj_weight, layer.multihead_attn.in_proj_bias
            # An image's beams attend to the same memory: fold them into the query axis, so the cross keys
            # and values are shared instead of copied per beam.
            q = F.linear(layer.norm2(x), w[:d], b[:d]).view(ck.shape[0], -1, ck.shape[1], ck.shape[3])
            a = F.scaled_dot_product_attention(q.transpose(1, 2), ck, cv)
            x = x + layer.multihead_attn.out_proj(a.transpose(1, 2).reshape(n, 1, d))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        return self.head(self.norm(x))[:, 0]

    @torch.no_grad()
    def generate(self, images, bos: int, eos: int, max_len: int | None = None, brace_delta=None, beams: int = 1):
        """Beam search (greedy for beams=1) -> (B, T) token ids starting with bos; finished rows are filled
        with eos. Beams are ranked by total log-probability.

        brace_delta: (vocab,) tensor of each token's net '{' minus '}'. With it, a token that would close a
        brace that isn't open is blocked, and a formula can't end while braces are open."""
        b, k, device = images.shape[0], beams, images.device
        n, steps = b * k, min(max_len or self.config["max_tokens"], self.config["max_tokens"])
        cross = self._cross_kv(self.encode(images))
        cache = [[None, None] for _ in self.decoder.layers]
        tokens = torch.full((n, steps), eos, dtype=torch.long, device=device)
        tokens[:, 0] = bos
        score = torch.zeros(b, k, device=device)
        score[:, 1:] = float("-inf")  # the beams start out identical: expand only the first
        depth = torch.zeros(n, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        first_beam = torch.arange(b, device=device)[:, None] * k
        t = -1
        for t in range(steps - 1):
            logp = self._step(tokens[:, t], t, cross, cache, steps).float().log_softmax(-1)
            if brace_delta is not None:
                logp[depth[:, None] + brace_delta[None] < 0] = float("-inf")
                logp[depth > 0, eos] = float("-inf")
            logp[done] = float("-inf")
            logp[done, eos] = 0.0  # a finished beam carries over unchanged
            vocab = logp.shape[1]
            score, idx = (score.view(n, 1) + logp).view(b, k * vocab).topk(k, -1)
            src, nxt = (first_beam + idx // vocab).view(-1), (idx % vocab).view(-1)
            tokens = tokens[src]
            tokens[:, t + 1] = nxt
            if k > 1:
                for kv in cache:
                    kv[0][:, :, :t + 1], kv[1][:, :, :t + 1] = kv[0][src, :, :t + 1], kv[1][src, :, :t + 1]
            if brace_delta is not None:
                depth = depth[src] + brace_delta[nxt]
            done = done[src] | (nxt == eos)
            if done.all():
                break
        best = tokens.view(b, k, steps)[torch.arange(b, device=device), score.argmax(-1)]
        return best[:, :t + 2]


def save_checkpoint(model: Im2Tex, path, **extra):
    """Model config + weights (+ e.g. optimizer state), written atomically so a crash keeps the old file."""
    tmp = f"{path}.tmp"
    torch.save({"config": model.config, "model": model.state_dict(), **extra}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cuda"):
    """-> (model in eval mode, full checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Im2Tex(**ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
