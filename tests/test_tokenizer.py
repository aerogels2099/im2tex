from im2tex.normalize import normalize
from im2tex.tokenizer import BOS, EOS, LINE, train

CORPUS = [normalize(f) for f in [
    r"\frac{a}{b} + x^2_i", r"\sum_{i=1}^n x_i", r"\text{for all } x \in \mathbb{R}",
    r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}", r"\hspace{0.5in} y", r"a \ b", r"\alpha\beta\gamma",
] * 20]


def test_round_trip():
    tok = train(CORPUS, vocab_size=300)
    for f in set(CORPUS):
        ids = tok.encode(f).ids
        assert normalize(tok.decode(ids)) == f


def test_frequent_tokens_are_atomic_and_unknown_words_still_encode():
    tok = train(CORPUS, vocab_size=300)
    enc = tok.encode(r"\frac { a } { b }")
    assert enc.tokens[0] == BOS and enc.tokens[-1] == EOS
    assert enc.tokens[1] == r"\frac</w>"
    unseen = r"\text {quantum} \frac { z } { q } \Xi"  # characters and words never seen in training
    assert normalize(tok.decode(tok.encode(unseen).ids)) == unseen
    assert tok.token_to_id("<unk>") not in tok.encode(unseen).ids


def test_line_token_comes_after_the_trained_vocabulary_and_text_round_trips():
    tok = train(CORPUS, vocab_size=300)
    assert tok.token_to_id(LINE) == tok.get_vocab_size() - 1 == tok.get_vocab_size(with_added_tokens=False)
    line = "Every matrix $ A = A ^ { T } $ is symmetric, isn't it?"
    assert tok.decode(tok.encode(line).ids) == line
