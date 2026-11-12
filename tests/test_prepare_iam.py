import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prepare_iam import detokenize  # noqa: E402


def test_detokenize_writes_iam_labels_back_as_written():
    assert detokenize("Griffiths, M P for Manchester Exchange .") == "Griffiths, M P for Manchester Exchange."
    assert detokenize("Nobody 's going to shove") == "Nobody's going to shove"
    assert detokenize('which would appear to " prop up " an out-dated') == 'which would appear to "prop up" an out-dated'
    assert detokenize("Party ( 280,000 members ) and") == "Party (280,000 members) and"
    assert detokenize("told them , ' the police will act . ' At five") == "told them, 'the police will act.' At five"


def test_detokenize_drops_crossed_out_words_and_keeps_year_apostrophes():
    assert detokenize("sentiment would # still favour") == "sentiment would still favour"
    assert detokenize("Since 1958 , 13 Labour life Peers and #") == "Since 1958, 13 Labour life Peers and"
    assert detokenize("that of the '30s . Shipbuilding") == "that of the '30s. Shipbuilding"
