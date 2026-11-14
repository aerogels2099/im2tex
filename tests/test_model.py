import pytest
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


def test_untrained_model_predicts_close_to_uniform():
    model = Im2Tex(vocab_size=3000, pretrained=False).eval()
    tokens = torch.randint(0, 3000, (2, 9))
    loss = torch.nn.functional.cross_entropy(model(torch.rand(2, 1, 64, 128), tokens).flatten(0, 1),
                                             tokens.flatten())
    assert abs(loss.item() - torch.log(torch.tensor(3000.0)).item()) < 1.0


def _until_eos(row):
    return row[:row.index(EOS) + 1] if EOS in row else row


def test_cached_step_matches_full_decoder():
    model = _tiny()
    with torch.no_grad():
        memory = model.encode(torch.rand(3, 1, 32, 64))
        tokens = torch.randint(0, 20, (3, 12))
        full = model.decode(memory, tokens)
        cross, cache = model._cross_kv(memory), [[None, None] for _ in model.decoder.layers]
        for t in range(12):
            assert torch.allclose(model._step(tokens[:, t], t, cross, cache, 12), full[:, t], atol=1e-5), t


def test_greedy_matches_rerunning_the_full_decoder():
    model = _tiny()
    images = torch.rand(4, 1, 32, 64)
    with torch.no_grad():  # the slow way: rerun the decoder over the whole prefix for every token
        memory, slow = model.encode(images), torch.full((4, 1), BOS)
        for _ in range(14):
            slow = torch.cat([slow, model.decode(memory, slow)[:, -1].argmax(-1, keepdim=True)], 1)
    fast = model.generate(images, BOS, EOS, max_len=15)
    assert [_until_eos(r) for r in fast.tolist()] == [_until_eos(r) for r in slow.tolist()]


def test_beam_search_keeps_images_apart():
    model = _tiny()
    images = torch.rand(3, 1, 32, 64)
    together = model.generate(images, BOS, EOS, beams=4)
    for i in range(3):
        alone = model.generate(images[i:i + 1], BOS, EOS, beams=4)[0]
        assert together[i, :len(alone)].tolist() == alone.tolist()
        assert (together[i, len(alone):] == EOS).all()


@pytest.mark.parametrize("beams", [1, 4])
def test_generate_never_closes_unopened_braces_or_ends_with_open_ones(beams):
    model = _tiny()
    with torch.no_grad():  # make braces likely, so the constraint actually has work to do
        model.head.weight[OPEN] *= 0
        model.head.weight[CLOSE] *= 0
        model.head.weight[OPEN] += 1.0
        model.head.weight[CLOSE] += 1.0
    delta = torch.zeros(20, dtype=torch.long)
    delta[OPEN], delta[CLOSE] = 1, -1
    out = model.generate(torch.rand(8, 1, 32, 64), BOS, EOS, brace_delta=delta, beams=beams)
    for row in out.tolist():
        depth = 0
        for t in row[1:]:
            if t == EOS:
                assert depth == 0
                break
            depth += {OPEN: 1, CLOSE: -1}.get(t, 0)
            assert depth >= 0


@pytest.mark.parametrize("beams", [1, 4])
def test_stop_loops_ends_a_repeating_reading_after_its_first_copy(beams):
    torch.manual_seed(0)
    model = Im2Tex(vocab_size=20, d_model=32, n_layers=1, n_heads=2, pretrained=False, max_tokens=200).eval()
    scripts = [[9, 8] + [5, 6, 7] * 100,  # a pattern repeating to the end
               [9] + [4] * 200,  # one token over and over
               [9, 3, EOS] + [4] * 200,  # ends normally
               [9, 8, 9, 8, 9, 8, 3, EOS] + [4] * 200]  # a short repeat, as in real formulas
    real_step = model._step

    def step(tokens, t, *args):  # the real step fills the cache beam search reorders; the scripted token wins
        logits = real_step(tokens, t, *args) * 0
        logits[torch.arange(tokens.shape[0]), [scripts[i // beams][t] for i in range(tokens.shape[0])]] = 50.0
        return logits
    model._step = step
    out, looped = model.generate(torch.rand(4, 1, 32, 64), BOS, EOS, beams=beams, stop_loops=True)
    assert [_until_eos(r) for r in out.tolist()] == [[BOS, 9, 8, 5, 6, 7, EOS], [BOS, 9, 4, EOS],
                                                     [BOS, 9, 3, EOS], [BOS, 9, 8, 9, 8, 9, 8, 3, EOS]]
    assert looped.tolist() == [True, True, False, False]


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


def test_grow_vocab_keeps_old_outputs_and_starts_new_tokens_as_a_copy():
    model = _tiny()
    x, t = torch.rand(1, 1, 32, 64), torch.randint(0, 20, (1, 5))
    before = model(x, t)
    model.grow_vocab(21, init_from=1)
    after = model(x, t)
    assert after.shape[-1] == 21 and model.config["vocab_size"] == 21
    assert torch.allclose(before, after[..., :20]) and torch.allclose(after[..., 20], after[..., 1])
    t2 = t.clone()
    t2[0, 0] = 20  # starting from the copy reads the same as starting from token 1
    t[0, 0] = 1
    assert torch.allclose(model(x, t2), model(x, t)) and model.head.weight is model.tok_emb.weight
