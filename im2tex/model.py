"""Image-to-LaTeX encoder-decoder: pretrained CNN encoder (stride 16) + transformer decoder."""
import os

import timm
import torch
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

    @torch.no_grad()
    def generate(self, images, bos: int, eos: int, max_len: int | None = None, brace_delta=None):
        """Greedy decoding -> (B, T) token ids starting with bos; finished rows are filled with eos.

        brace_delta: (vocab,) tensor of each token's net '{' minus '}'. With it, a token that would close a
        brace that isn't open is blocked, and a formula can't end while braces are open."""
        memory = self.encode(images)
        b, device = images.shape[0], images.device
        tokens = torch.full((b, 1), bos, dtype=torch.long, device=device)
        depth = torch.zeros(b, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(min(max_len or self.config["max_tokens"], self.config["max_tokens"]) - 1):
            logits = self.decode(memory, tokens)[:, -1].float()  # no KV cache yet: recomputes the prefix
            if brace_delta is not None:
                logits[depth[:, None] + brace_delta[None] < 0] = float("-inf")
                logits[depth > 0, eos] = float("-inf")
            nxt = torch.where(done, eos, logits.argmax(-1))
            tokens = torch.cat([tokens, nxt[:, None]], 1)
            if brace_delta is not None:
                depth += brace_delta[nxt]
            done |= nxt == eos
            if done.all():
                break
        return tokens


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
