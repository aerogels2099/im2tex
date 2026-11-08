"""Model tokenizer: one token per normalized LaTeX token (\\frac, {, x), BPE pieces only for rare ones.

Normalized formulas are single-space separated, so splitting on whitespace and decoding with spaces
round-trips them (up to the spacing around \\ , which renders the same).
"""
import itertools

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
ASCII = [chr(c) for c in range(33, 127)]


def train(formulas, vocab_size: int = 4000) -> Tokenizer:
    """Train on normalized formulas (an iterable of strings)."""
    tok = Tokenizer(models.BPE(unk_token=UNK, end_of_word_suffix="</w>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.decoder = decoders.BPEDecoder(suffix="</w>")
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[PAD, BOS, EOS, UNK],
                                  end_of_word_suffix="</w>", initial_alphabet=ASCII, show_progress=False)
    # One extra line with every ASCII character as its own word, so any of them can end a word too.
    tok.train_from_iterator(itertools.chain(formulas, [" ".join(ASCII)]), trainer)
    tok.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A {EOS}", special_tokens=[(BOS, tok.token_to_id(BOS)), (EOS, tok.token_to_id(EOS))])
    return tok


def load(path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def brace_delta(tok: Tokenizer) -> list[int]:
    """Each token id's net '{' minus '}' (escaped \\{ and \\} don't count), for constrained decoding."""
    delta = [0] * tok.get_vocab_size()
    for piece, i in tok.get_vocab().items():
        s = piece.removesuffix("</w>").replace("\\{", "").replace("\\}", "")
        delta[i] = s.count("{") - s.count("}")
    return delta
