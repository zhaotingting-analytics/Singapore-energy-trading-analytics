"""
Custom BiLSTM + Attention sentiment classifier for oil-market news.

Architecture
────────────
  Embedding (trainable)
       ↓
  Dropout
       ↓
  Bi-LSTM  (2 layers)
       ↓
  Additive Attention  ← learns which words matter most
       ↓
  MLP classifier (3 classes: negative / neutral / positive)

Training
────────
  No labelled dataset is required out-of-the-box.
  SilverLabelBootstrapper uses VADER to auto-label a corpus, then trains
  the LSTM on those weak labels — the neural model then generalises better
  to financial terminology that VADER rules miss.

  Call  NeuralSentimentTrainer.train(articles)  to bootstrap.

Usage
─────
  analyzer = NeuralSentimentAnalyzer()
  analyzer.fit(articles)          # train from scratch (or load from disk)
  score = analyzer.score(text)    # returns float in [-1, +1]
"""

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser
# ─────────────────────────────────────────────────────────────────────────────

class FinancialTokenizer:
    """
    Word-level tokeniser tuned for financial text.

    Preserves dollar amounts ($80.5), percentages (2.3%), and common
    abbreviations (bpd, WTI, OPEC) as single tokens.
    """

    PAD_IDX = 0
    UNK_IDX = 1
    _SPECIAL = {"<PAD>": 0, "<UNK>": 1}

    def __init__(self, max_vocab: int = 12_000):
        self.max_vocab = max_vocab
        self.word2idx: Dict[str, int] = dict(self._SPECIAL)
        self.idx2word: Dict[int, str] = {v: k for k, v in self._SPECIAL.items()}
        self._fitted = False

    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # keep $XX.X price tokens and N% percentage tokens intact
        text = re.sub(r"\$\s*(\d+\.?\d*)", r"$\1", text)
        text = re.sub(r"(\d+\.?\d*)\s*%", r"\1%", text)
        # strip non-alphanumeric except $, %, +, -, and dots between digits
        text = re.sub(r"[^a-z0-9\$%\+\-\s\.]", " ", text)
        # remove dots that are NOT between two digits (sentence dots, etc.)
        text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
        return [t for t in text.split() if t]

    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> "FinancialTokenizer":
        counter: Counter = Counter()
        for t in texts:
            counter.update(self._tokenize(t))
        for word, _ in counter.most_common(self.max_vocab - len(self._SPECIAL)):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self._fitted = True
        log.info("Vocabulary built: %d tokens", len(self.word2idx))
        return self

    # ------------------------------------------------------------------
    def encode(self, text: str, max_len: int = 128) -> List[int]:
        return [
            self.word2idx.get(t, self.UNK_IDX)
            for t in self._tokenize(text)[:max_len]
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: str) -> "FinancialTokenizer":
        obj = cls()
        with open(path, encoding="utf-8") as f:
            obj.word2idx = json.load(f)
        obj.idx2word = {v: k for k, v in obj.word2idx.items()}
        obj._fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch model
# ─────────────────────────────────────────────────────────────────────────────

class _AdditiveAttention(nn.Module):
    """Single-query additive (Bahdanau-style) attention over LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # lstm_out : (B, T, H)
        weights = self.score(lstm_out).squeeze(-1)          # (B, T)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = F.softmax(weights, dim=-1)                # (B, T)
        return (lstm_out * weights.unsqueeze(-1)).sum(dim=1) # (B, H)


class BiLSTMAttentionModel(nn.Module):
    """
    Bidirectional LSTM with additive attention.

    Args:
        vocab_size : vocabulary size (from tokeniser)
        embed_dim  : word embedding dimension
        hidden_dim : LSTM hidden state dimension (per direction)
        num_layers : number of LSTM layers
        dropout    : dropout rate applied after embeddings and between LSTM layers
        num_classes: 3 (negative=0, neutral=1, positive=2)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
        num_classes: int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _AdditiveAttention(hidden_dim * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_ids      : (B, T)
        # attention_mask : (B, T)  1 for real tokens, 0 for padding
        x = self.embed_drop(self.embedding(input_ids))   # (B, T, E)
        lstm_out, _ = self.lstm(x)                        # (B, T, H*2)
        ctx = self.attention(lstm_out, attention_mask)    # (B, H*2)
        return self.head(ctx)                             # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _LabelledSample:
    ids:   List[int]
    label: int   # 0=negative, 1=neutral, 2=positive


class _SentimentDataset(Dataset):
    def __init__(self, samples: List[_LabelledSample], max_len: int = 128):
        self.samples = samples
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        ids = s.ids[: self.max_len]
        pad_len = self.max_len - len(ids)
        ids_padded = ids + [0] * pad_len
        mask = [1] * len(ids) + [0] * pad_len
        return {
            "input_ids":      torch.tensor(ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask,       dtype=torch.long),
            "labels":         torch.tensor(s.label,    dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Silver-label bootstrapper
# ─────────────────────────────────────────────────────────────────────────────

def _vader_label(text: str) -> Optional[int]:
    """Return 0/1/2 using VADER compound score; None on import failure."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        score = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        if score >= 0.05:
            return 2   # positive
        if score <= -0.05:
            return 0   # negative
        return 1       # neutral
    except ImportError:
        return None


# Seed corpus: manually labelled financial headlines used when the
# provided article list is too small (< 30 samples).
_SEED_CORPUS: List[Tuple[str, int]] = [
    # positive (label=2)
    ("OPEC announces surprise supply cut boosting oil prices above $90", 2),
    ("Strong demand recovery lifts Brent crude to six-month high", 2),
    ("Oil surges as Saudi Arabia pledges extended production cuts", 2),
    ("IEA upgrades global oil demand outlook, prices rally sharply", 2),
    ("WTI crude posts best weekly gain in three months on supply fears", 2),
    ("Robust refinery margins push petroleum stocks higher across Asia", 2),
    ("Energy sector outperforms market as Brent exceeds $88 per barrel", 2),
    ("Gasoline demand hits record high as summer driving season peaks", 2),
    # negative (label=0)
    ("Oil prices plunge on fears of global recession and demand collapse", 0),
    ("Crude oil tumbles after unexpected US inventory build", 0),
    ("WTI falls below $70 as economic slowdown signals weigh on market", 0),
    ("OPEC+ faces internal rift threatening to unwind production agreement", 0),
    ("Oil prices sink on weak Chinese factory data raising demand concerns", 0),
    ("Rising interest rates crush energy demand forecasts, crude drops 4%", 0),
    ("Brent crude plunges after Iran nuclear deal progress dampens supply fears", 0),
    ("Refinery outages ease, oil inventory surplus pressures prices lower", 0),
    # neutral (label=1)
    ("Oil markets steady as traders await next OPEC policy meeting", 1),
    ("Crude prices little changed ahead of US Federal Reserve decision", 1),
    ("WTI holds near $80 after mixed inventory data", 1),
    ("Oil trades sideways as geopolitical risks offset demand concerns", 1),
    ("Crude benchmark unchanged despite surprise output figures from Russia", 1),
    ("Energy markets pause as investors digest OPEC communique", 1),
    ("Brent flat amid competing signals on global growth trajectory", 1),
]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSentimentTrainer:
    """
    Train (or fine-tune) the BiLSTM model on oil-news text.

    Steps:
      1. Collect texts from `articles` and the seed corpus.
      2. Auto-label the articles with VADER (silver labels).
      3. Build vocabulary.
      4. Train BiLSTMAttentionModel with cross-entropy loss.
      5. Save tokeniser + model weights to `save_dir`.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
        epochs: int = 15,
        batch_size: int = 16,
        lr: float = 1e-3,
        max_len: int = 128,
        save_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_len = max_len
        self.save_dir = save_dir or os.path.join("output", "neural_model")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # ------------------------------------------------------------------
    def _build_samples(self, extra_texts: List[str]) -> List[_LabelledSample]:
        samples: List[_LabelledSample] = []

        # Seed corpus
        for text, label in _SEED_CORPUS:
            ids = self._tokenizer.encode(text, self.max_len)
            samples.append(_LabelledSample(ids=ids, label=label))

        # Silver-labelled article texts
        for text in extra_texts:
            label = _vader_label(text)
            if label is None:
                continue
            ids = self._tokenizer.encode(text, self.max_len)
            samples.append(_LabelledSample(ids=ids, label=label))

        return samples

    # ------------------------------------------------------------------
    def train(self, articles=None) -> "NeuralSentimentTrainer":
        """
        Bootstrap-train from articles + seed corpus.

        Args:
            articles: list of NewsArticle objects (optional).
                      Passing None trains on the seed corpus only.
        """
        extra_texts: List[str] = []
        if articles:
            extra_texts = [a.combined_text for a in articles]

        all_texts = [t for t, _ in _SEED_CORPUS] + extra_texts
        self._tokenizer = FinancialTokenizer()
        self._tokenizer.fit(all_texts)

        samples = self._build_samples(extra_texts)
        log.info("Training on %d samples (device=%s)", len(samples), self.device)

        self._model = BiLSTMAttentionModel(
            vocab_size=self._tokenizer.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        dataset = _SentimentDataset(samples, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimiser = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            correct = 0
            for batch in loader:
                ids   = batch["input_ids"].to(self.device)
                mask  = batch["attention_mask"].to(self.device)
                label = batch["labels"].to(self.device)

                optimiser.zero_grad()
                logits = self._model(ids, mask)
                loss   = criterion(logits, label)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimiser.step()

                total_loss += loss.item() * len(label)
                correct    += (logits.argmax(-1) == label).sum().item()

            scheduler.step()
            avg_loss = total_loss / len(samples)
            acc      = correct / len(samples)
            if epoch % 5 == 0 or epoch == 1:
                log.info("Epoch %2d/%d  loss=%.4f  acc=%.3f", epoch, self.epochs, avg_loss, acc)

        self._save()
        return self

    # ------------------------------------------------------------------
    def _save(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        tok_path   = os.path.join(self.save_dir, "tokenizer.json")
        model_path = os.path.join(self.save_dir, "bilstm_weights.pt")
        cfg_path   = os.path.join(self.save_dir, "model_config.json")

        self._tokenizer.save(tok_path)
        torch.save(self._model.state_dict(), model_path)
        with open(cfg_path, "w") as f:
            json.dump({
                "vocab_size": self._tokenizer.vocab_size,
                "embed_dim":  self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout":    self.dropout,
            }, f)
        log.info("Model saved to %s", self.save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSentimentAnalyzer:
    """
    Drop-in inference interface for the trained BiLSTM model.

    score(text) → float in [-1, +1]
      = P(positive) − P(negative)

    Lifecycle:
      1. Call fit(articles) to train from scratch and cache to disk.
         OR: if a saved model exists at model_dir, it is auto-loaded.
      2. Call score(text) for inference.
    """

    # Class-level index → score mapping:  positive=2 → +1, negative=0 → -1, neutral=1 → 0
    _IDX_TO_SCALAR = {0: -1.0, 1: 0.0, 2: 1.0}

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        trainer_kwargs: Optional[dict] = None,
    ):
        self.model_dir = model_dir or os.path.join("output", "neural_model")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._trainer_kwargs = trainer_kwargs or {}
        self._model:     Optional[BiLSTMAttentionModel] = None
        self._tokenizer: Optional[FinancialTokenizer]   = None

        # Auto-load from disk if available
        if self._saved_model_exists():
            self._load()

    # ------------------------------------------------------------------
    def _saved_model_exists(self) -> bool:
        return (
            os.path.exists(os.path.join(self.model_dir, "bilstm_weights.pt"))
            and os.path.exists(os.path.join(self.model_dir, "tokenizer.json"))
            and os.path.exists(os.path.join(self.model_dir, "model_config.json"))
        )

    # ------------------------------------------------------------------
    def _load(self) -> None:
        cfg_path   = os.path.join(self.model_dir, "model_config.json")
        tok_path   = os.path.join(self.model_dir, "tokenizer.json")
        model_path = os.path.join(self.model_dir, "bilstm_weights.pt")

        with open(cfg_path) as f:
            cfg = json.load(f)

        self._tokenizer = FinancialTokenizer.load(tok_path)
        self._model = BiLSTMAttentionModel(**cfg).to(self.device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self._model.eval()
        log.info("BiLSTM model loaded from %s", self.model_dir)

    # ------------------------------------------------------------------
    def fit(self, articles=None) -> "NeuralSentimentAnalyzer":
        """Train from scratch using articles + seed corpus, then cache."""
        kwargs = {"save_dir": self.model_dir, **self._trainer_kwargs}
        trainer = NeuralSentimentTrainer(**kwargs)
        trainer.train(articles)
        self._model     = trainer._model
        self._tokenizer = trainer._tokenizer
        self._model.eval()
        return self

    # ------------------------------------------------------------------
    def score(self, text: str) -> Tuple[float, str]:
        """
        Returns (composite_score, label).
          composite_score : P(positive) − P(negative), range [-1, +1]
          label           : "positive" / "negative" / "neutral"
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call .fit(articles) first, "
                "or provide a pre-trained model_dir."
            )
        ids = self._tokenizer.encode(text, max_len=128)
        if not ids:
            return 0.0, "neutral"

        tensor = torch.tensor([ids], dtype=torch.long).to(self.device)
        mask   = torch.ones_like(tensor)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(tensor, mask)          # (1, 3)
            probs  = F.softmax(logits, dim=-1).squeeze() # (3,)

        p_neg, p_neu, p_pos = probs[0].item(), probs[1].item(), probs[2].item()
        composite = p_pos - p_neg  # range [-1, +1]

        if composite >= 0.05:
            label = "positive"
        elif composite <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return composite, label

    # ------------------------------------------------------------------
    def attention_weights(self, text: str) -> List[Tuple[str, float]]:
        """
        Return (token, attention_weight) pairs for interpretability.
        Useful for understanding which words drove the prediction.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded.")

        tokens_raw = re.sub(r"[^a-z0-9\$%\+\-\s]", " ", text.lower()).split()
        ids = self._tokenizer.encode(text, max_len=128)
        tokens = tokens_raw[: len(ids)]

        tensor = torch.tensor([ids], dtype=torch.long).to(self.device)
        mask   = torch.ones_like(tensor)

        self._model.eval()
        with torch.no_grad():
            emb      = self._model.embed_drop(self._model.embedding(tensor))
            lstm_out, _ = self._model.lstm(emb)
            scores   = self._model.attention.score(lstm_out).squeeze()
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            weights  = F.softmax(scores, dim=-1).cpu().numpy()

        return list(zip(tokens, weights.tolist()))
