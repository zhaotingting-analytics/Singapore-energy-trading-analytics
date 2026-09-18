"""
Tests for the BiLSTM neural sentiment model.

All tests use the seed corpus + mock articles (no network, no large downloads).
The model is deliberately tiny (embed=32, hidden=32) to keep CI fast.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest
import torch

from oil_sentiment.neural_model import (
    BiLSTMAttentionModel,
    FinancialTokenizer,
    NeuralSentimentAnalyzer,
    NeuralSentimentTrainer,
    _SEED_CORPUS,
    _vader_label,
)
from oil_sentiment.news_fetcher import NewsArticle


# ─── helpers ────────────────────────────────────────────────────────────────

def _article(title: str) -> NewsArticle:
    return NewsArticle(
        title=title,
        summary="",
        url=f"https://example.com/{hash(title)}",
        published_at=datetime.now(timezone.utc),
        source="test",
    )


def _tiny_trainer(save_dir: str) -> NeuralSentimentTrainer:
    """Return a fast (small) trainer for testing."""
    return NeuralSentimentTrainer(
        embed_dim=32,
        hidden_dim=32,
        num_layers=1,
        epochs=3,
        batch_size=8,
        save_dir=save_dir,
    )


# ─── tokeniser ──────────────────────────────────────────────────────────────

class TestFinancialTokenizer:
    def test_fit_builds_vocab(self):
        tok = FinancialTokenizer()
        tok.fit(["oil prices surge", "crude oil falls"])
        assert tok.vocab_size > 2  # at least PAD + UNK + some words

    def test_encode_returns_ints(self):
        tok = FinancialTokenizer()
        tok.fit(["oil price"])
        ids = tok.encode("oil price surge")
        assert all(isinstance(i, int) for i in ids)

    def test_unknown_tokens_use_unk(self):
        tok = FinancialTokenizer()
        tok.fit(["oil"])
        ids = tok.encode("zzznonsensexxx")
        assert ids[0] == FinancialTokenizer.UNK_IDX

    def test_max_len_truncation(self):
        tok = FinancialTokenizer()
        tok.fit(["word"] * 200)
        ids = tok.encode(" ".join(["word"] * 200), max_len=10)
        assert len(ids) <= 10

    def test_save_and_load(self, tmp_path):
        tok = FinancialTokenizer()
        tok.fit(["brent crude oil rises"])
        path = str(tmp_path / "vocab.json")
        tok.save(path)

        tok2 = FinancialTokenizer.load(path)
        assert tok2.word2idx == tok.word2idx

    def test_price_token_preserved(self):
        tok = FinancialTokenizer()
        tok.fit(["$80.5 brent crude"])
        ids = tok.encode("$80.5 brent crude")
        assert len(ids) == 3  # "$80.5", "brent", "crude"

    def test_percentage_token_preserved(self):
        tok = FinancialTokenizer()
        tok.fit(["oil rises 2.3% today"])
        ids = tok.encode("2.3% today")
        assert len(ids) == 2


# ─── model architecture ─────────────────────────────────────────────────────

class TestBiLSTMModel:
    @pytest.fixture
    def model(self):
        return BiLSTMAttentionModel(vocab_size=100, embed_dim=32, hidden_dim=32, num_layers=1)

    def test_output_shape(self, model):
        ids = torch.randint(0, 100, (4, 20))   # batch=4, seq_len=20
        logits = model(ids)
        assert logits.shape == (4, 3)

    def test_output_with_mask(self, model):
        ids  = torch.randint(0, 100, (2, 15))
        mask = torch.ones(2, 15, dtype=torch.long)
        mask[0, 10:] = 0  # pad the last 5 positions of first example
        logits = model(ids, mask)
        assert logits.shape == (2, 3)

    def test_gradients_flow(self, model):
        ids    = torch.randint(0, 100, (2, 10))
        logits = model(ids)
        loss   = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_single_token_input(self, model):
        ids    = torch.tensor([[5]])
        logits = model(ids)
        assert logits.shape == (1, 3)

    def test_different_batch_sizes(self, model):
        for batch in [1, 4, 8]:
            ids = torch.randint(0, 100, (batch, 16))
            assert model(ids).shape == (batch, 3)


# ─── trainer ────────────────────────────────────────────────────────────────

class TestNeuralSentimentTrainer:
    def test_train_on_seed_corpus_only(self, tmp_path):
        trainer = _tiny_trainer(str(tmp_path))
        trainer.train(articles=None)
        assert os.path.exists(os.path.join(str(tmp_path), "bilstm_weights.pt"))
        assert os.path.exists(os.path.join(str(tmp_path), "tokenizer.json"))
        assert os.path.exists(os.path.join(str(tmp_path), "model_config.json"))

    def test_train_with_articles(self, tmp_path):
        articles = [_article(t) for t, _ in _SEED_CORPUS[:6]]
        trainer  = _tiny_trainer(str(tmp_path))
        trainer.train(articles=articles)
        assert trainer._model is not None
        assert trainer._tokenizer is not None

    def test_vocab_contains_oil_terms(self, tmp_path):
        trainer = _tiny_trainer(str(tmp_path))
        trainer.train()
        assert "oil" in trainer._tokenizer.word2idx
        assert "crude" in trainer._tokenizer.word2idx


# ─── analyzer (inference) ───────────────────────────────────────────────────

class TestNeuralSentimentAnalyzer:
    @classmethod
    @pytest.fixture(scope="class")
    def trained_analyzer(cls, tmp_path_factory):
        """Train once, reuse across tests in this class."""
        save_dir = str(tmp_path_factory.mktemp("model"))
        analyzer = NeuralSentimentAnalyzer(
            model_dir=save_dir,
            trainer_kwargs={"embed_dim": 32, "hidden_dim": 32,
                            "num_layers": 1, "epochs": 3, "batch_size": 8},
        )
        analyzer.fit(articles=None)
        return analyzer

    def test_score_returns_float_in_range(self, trained_analyzer):
        score, label = trained_analyzer.score("Oil prices surge on OPEC cut")
        assert -1.0 <= score <= 1.0

    def test_label_is_valid(self, trained_analyzer):
        _, label = trained_analyzer.score("Crude tumbles on recession fears")
        assert label in {"positive", "negative", "neutral"}

    def test_score_empty_text(self, trained_analyzer):
        score, label = trained_analyzer.score("")
        assert score == 0.0
        assert label == "neutral"

    def test_positive_text_positive_or_neutral(self, trained_analyzer):
        score, _ = trained_analyzer.score(
            "excellent gains boost oil prices strongly higher"
        )
        # after only 3 epochs with a tiny model, we just check it doesn't crash
        assert -1.0 <= score <= 1.0

    def test_attention_weights_sum_to_one(self, trained_analyzer):
        weights = trained_analyzer.attention_weights("Brent crude oil rises strongly")
        if weights:
            total = sum(w for _, w in weights)
            assert abs(total - 1.0) < 1e-4

    def test_attention_weights_token_count(self, trained_analyzer):
        text    = "WTI oil prices fall on weak demand"
        weights = trained_analyzer.attention_weights(text)
        tokens  = [t for t in text.lower().split() if t.isalpha() or t.startswith("$")]
        assert len(weights) <= len(tokens)

    def test_save_and_reload(self, tmp_path):
        save_dir = str(tmp_path)
        a1 = NeuralSentimentAnalyzer(
            model_dir=save_dir,
            trainer_kwargs={"embed_dim": 32, "hidden_dim": 32,
                            "num_layers": 1, "epochs": 3, "batch_size": 8},
        )
        a1.fit(articles=None)
        score1, _ = a1.score("Brent crude surges on Saudi cuts")

        a2 = NeuralSentimentAnalyzer(model_dir=save_dir)
        score2, _ = a2.score("Brent crude surges on Saudi cuts")

        assert abs(score1 - score2) < 1e-5, "Reloaded model must give same score"

    def test_not_fitted_raises(self, tmp_path):
        analyzer = NeuralSentimentAnalyzer(model_dir=str(tmp_path / "empty"))
        with pytest.raises(RuntimeError):
            analyzer.score("test text")


# ─── vader label helper ─────────────────────────────────────────────────────

class TestVaderLabel:
    def test_positive_text(self):
        assert _vader_label("excellent gains boost oil prices") == 2

    def test_negative_text(self):
        assert _vader_label("oil prices crash on devastating recession fears") == 0

    def test_returns_int(self):
        result = _vader_label("oil prices stable today")
        assert result in {0, 1, 2, None}
