from __future__ import annotations

from typing import Any

import pytest

from rt import embed


class FakeSentenceTransformer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        return [1]


def test_text_embedder_does_not_compile_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls = []

    monkeypatch.setattr(embed, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        embed.torch,
        "compile",
        lambda model: compile_calls.append(model) or model,
    )

    embed.TextEmbedder(
        batch_size=1,
        embedding_model="all-MiniLM-L12-v2",
        device_type="cpu",
        compile_model=False,
    )

    assert compile_calls == []


def test_text_embedder_compiles_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls = []

    monkeypatch.setattr(embed, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        embed.torch,
        "compile",
        lambda model: compile_calls.append(model) or model,
    )

    embed.TextEmbedder(
        batch_size=1,
        embedding_model="all-MiniLM-L12-v2",
        device_type="cuda",
        compile_model=True,
    )

    assert len(compile_calls) == 1
