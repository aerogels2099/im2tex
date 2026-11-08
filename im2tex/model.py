"""Image-to-LaTeX encoder-decoder: pretrained CNN encoder (stride 16) + transformer decoder."""
import timm
import torch
from torch import nn


class Im2Tex(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8,
                 encoder: str = "convnext_tiny", pretrained: bool = True,
                 max_tokens: int = 512, max_rows: int = 64, max_cols: int = 256):
        super().__init__()
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

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H, W) with H, W multiples of 16 -> memory (B, H/16 * W/16, d_model)."""
        f = self.encoder(images)[-1]
        _, _, h, w = f.shape
        pos = self.row_emb.weight[:h, None] + self.col_emb.weight[None, :w]
        return self.proj(f.flatten(2).transpose(1, 2)) + pos.reshape(1, h * w, -1)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits (B, T, vocab) for decoder inputs tokens (B, T)."""
        memory = self.encode(images)
        t = tokens.shape[1]
        x = self.tok_emb(tokens) + self.pos_emb.weight[:t]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=tokens.device, dtype=x.dtype)
        x = self.decoder(x, memory, tgt_mask=mask, tgt_is_causal=True)
        return self.head(self.norm(x))
