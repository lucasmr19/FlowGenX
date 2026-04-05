import pytest
from src.representations.sequential.tokenizer import TokenVocabulary


def test_special_tokens_present():
    vocab = TokenVocabulary(max_vocab_size=100)

    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    for token in specials:
        assert token in vocab._token2id


def test_add_token():
    vocab = TokenVocabulary(max_vocab_size=100)
    vocab.add_token("ip_proto:6")

    assert "ip_proto:6" in vocab._token2id
    assert vocab._token2id["ip_proto:6"] >= 0


def test_vocab_max_size_limit():
    vocab = TokenVocabulary(max_vocab_size=100)

    initial_size = len(vocab)

    for i in range(20):
        vocab.add_token(f"t{i}")

    assert len(vocab) <= vocab.max_vocab_size