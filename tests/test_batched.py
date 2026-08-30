# -*- coding: utf-8 -*-
"""Deterministic unit tests for lazy batching and batch-size validation.

These tests do not load any embedding model, PDF, or vector store.
"""
from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest
from langchain_core.documents import Document

from src.retriever import _batched, validate_batch_size


def _doc(cid: str, text: str = "x") -> Document:
    return Document(page_content=text, metadata={"chunk_id": cid})


# --------------------------------------------------------------------------- #
# validate_batch_size
# --------------------------------------------------------------------------- #

class TestValidateBatchSize:
    def test_positive_integer_passes_through(self) -> None:
        assert validate_batch_size(1) == 1
        assert validate_batch_size(128) == 128

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_non_positive_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            validate_batch_size(bad)

    @pytest.mark.parametrize("bad", [3.0, "8", None, [4]])
    def test_non_integer_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            validate_batch_size(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [True, False])
    def test_bool_rejected(self, bad: bool) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            validate_batch_size(bad)


# --------------------------------------------------------------------------- #
# _batched — laziness
# --------------------------------------------------------------------------- #

class TestBatchedLaziness:
    def test_returns_generator_not_list(self) -> None:
        chunks = [_doc(str(i)) for i in range(5)]
        result = _batched(chunks, 2)
        assert inspect.isgenerator(result)

    def test_is_iterator(self) -> None:
        chunks = [_doc(str(i)) for i in range(5)]
        result = _batched(chunks, 2)
        assert isinstance(result, Iterator)

    def test_len_raises_type_error(self) -> None:
        """A generator has no len — proves we did not materialise a list."""
        chunks = [_doc(str(i)) for i in range(5)]
        result = _batched(chunks, 2)
        with pytest.raises(TypeError):
            len(result)  # type: ignore[arg-type]

    def test_does_not_materialise_all_batches(self) -> None:
        """Consuming only the first batch must not exhaust the generator."""
        chunks = [_doc(str(i)) for i in range(6)]
        gen = _batched(chunks, 2)
        first = next(gen)
        assert len(first) == 2
        # remaining batches still available
        rest = list(gen)
        assert len(rest) == 2
        assert len(rest[0]) == 2
        assert len(rest[1]) == 2


# --------------------------------------------------------------------------- #
# _batched — batch boundaries
# --------------------------------------------------------------------------- #

class TestBatchedBoundaries:
    def test_exact_multiple(self) -> None:
        chunks = [_doc(str(i)) for i in range(6)]
        batches = list(_batched(chunks, 3))
        assert len(batches) == 2
        assert all(len(b) == 3 for b in batches)

    def test_remainder_batch(self) -> None:
        chunks = [_doc(str(i)) for i in range(7)]
        batches = list(_batched(chunks, 3))
        assert len(batches) == 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1

    def test_single_batch_when_batch_size_exceeds_len(self) -> None:
        chunks = [_doc(str(i)) for i in range(3)]
        batches = list(_batched(chunks, 100))
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_empty_input_yields_no_batches(self) -> None:
        batches = list(_batched([], 10))
        assert batches == []

    def test_batch_size_one(self) -> None:
        chunks = [_doc("a"), _doc("b"), _doc("c")]
        batches = list(_batched(chunks, 1))
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_preserves_chunk_identity(self) -> None:
        chunks = [_doc(f"c{i}", text=f"text{i}") for i in range(4)]
        batches = list(_batched(chunks, 2))
        assert batches[0][0].page_content == "text0"
        assert batches[1][1].page_content == "text3"

    def test_batch_boundaries_with_large_count(self) -> None:
        chunks = [_doc(str(i)) for i in range(100)]
        batches = list(_batched(chunks, 30))
        assert len(batches) == 4
        assert [len(b) for b in batches] == [30, 30, 30, 10]


# --------------------------------------------------------------------------- #
# _batched — invalid batch size
# --------------------------------------------------------------------------- #

class TestBatchedInvalidSize:
    @pytest.mark.parametrize("bad", [0, -1, -5])
    def test_non_positive_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            list(_batched([_doc("a")], bad))

    def test_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            list(_batched([_doc("a")], True))  # type: ignore[arg-type]

    def test_float_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            list(_batched([_doc("a")], 2.0))  # type: ignore[arg-type]
