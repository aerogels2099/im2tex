import torch

from im2tex.model import Im2Tex, load_checkpoint, save_checkpoint
from im2tex.tokenizer import brace_delta, train

BOS, EOS, OPEN, CLOSE = 1, 2, 5, 6


def _tiny():
    torch.manual_seed(0)
    return Im2Tex(vocab_size=20, d_model=32, n_layers=1, n_heads=2, pretrained=False, max_tokens=40).eval()


def test_forward_shapes():
    logits = _tiny()(torch.rand(2, 1, 32, 64), torch.randint(0, 20, (2, 7)))
    assert logits.shape == (2, 7, 20)


def test_generate_never_closes_unopened_braces_or_ends_with_open_ones():
    model = _tiny()
    with torch.no_grad():  # make braces likely, so the constraint actually has work to do
        model.head.weight[OPEN] *= 0
        model.head.weight[CLOSE] *= 0
        model.head.weight[OPEN] += 1.0
        model.head.weight[CLOSE] += 1.0
    delta = torch.zeros(20, dtype=torch.long)
    delta[OPEN], delta[CLOSE] = 1, -1
    out = model.generate(torch.rand(8, 1, 32, 64), BOS, EOS, brace_delta=delta)
    for row in out.tolist():
        depth = 0
        for t in row[1:]:
            if t == EOS:
                assert depth == 0
                break
            depth += {OPEN: 1, CLOSE: -1}.get(t, 0)
            assert depth >= 0


def test_checkpoint_round_trip(tmp_path):
    model = _tiny()
    save_checkpoint(model, tmp_path / "m.pt", step=7)
    loaded, ckpt = load_checkpoint(tmp_path / "m.pt", device="cpu")
    x, t = torch.rand(1, 1, 32, 64), torch.randint(0, 20, (1, 5))
    assert ckpt["step"] == 7 and torch.allclose(model(x, t), loaded(x, t))


def test_brace_delta_counts_braces_but_not_escaped_ones():
    tok = train([r"\frac { a } { b } \{ x \} \text {if }"] * 5, vocab_size=200)
    d = brace_delta(tok)
    assert d[tok.token_to_id("{</w>")] == 1 and d[tok.token_to_id("}</w>")] == -1
    for s, balance in [(r"\frac { a } { b }", 0), (r"\{ x \}", 0), (r"\text {if }", 0), (r"\text {if", 1)]:
        assert sum(d[i] for i in tok.encode(s).ids) == balance, s
