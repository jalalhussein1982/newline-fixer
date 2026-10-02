import torch

from newline_fixer.models.scratch_config import TINY, ScratchConfig
from newline_fixer.models.scratch_net import CharCNN, GapTagger, count_parameters
from newline_fixer.models.vocab import NO_GAP, Batch


def make_batch(lengths: list[int], max_chars: int = 8) -> Batch:
    b, n = len(lengths), max(lengths)
    g = torch.full((b, n), NO_GAP, dtype=torch.long)
    for i, length in enumerate(lengths):
        g[i, : length - 1] = 1
    return Batch(
        word=torch.randint(2, 50, (b, n)),
        char=torch.randint(2, 20, (b, n, max_chars)),
        gap_after=g,
        lengths=torch.tensor(lengths),
    )


def test_char_cnn_shape() -> None:
    cnn = CharCNN(n_chars=20, char_dim=4, filters=6, width=3)
    out = cnn(torch.randint(0, 20, (2, 5, 8)))
    assert out.shape == (2, 5, 6)


def test_gap_tagger_logits_shape_and_two_token_window() -> None:
    net = GapTagger(TINY, n_words=50, n_chars=20)
    logits = net(make_batch([5, 2]))
    assert logits.shape == (2, 4, 4)
    single = net(make_batch([2]))
    assert single.shape == (1, 1, 4)


def test_gradients_reach_every_parameter() -> None:
    net = GapTagger(TINY, n_words=50, n_chars=20)
    logits = net(make_batch([4, 3]))
    logits.sum().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert missing == []


def test_full_config_is_about_five_million_parameters() -> None:
    net = GapTagger(ScratchConfig(), n_words=30_002, n_chars=300)
    n = count_parameters(net)
    assert 4_500_000 < n < 7_000_000
