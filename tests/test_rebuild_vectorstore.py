# -*- coding: utf-8 -*-
"""Deterministic unit tests for scripts/rebuild_vectorstore.py.

These tests do not load any embedding model, PDF, or vector store.
They verify argparse defaults, mode resolution, and incremental/full sync
argument wiring via mocks.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the script via importlib so we don't need scripts/__init__.py.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "rebuild_vectorstore.py"
_spec = importlib.util.spec_from_file_location("rebuild_vectorstore", _SCRIPT_PATH)
rebuild = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rebuild)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse(*argv: str) -> argparse.Namespace:
    return rebuild.build_parser().parse_args(argv)


def _fake_idx_result(chunks=None, changed=None, deleted=None, full_rebuild=False):
    """Return a simple namespace mimicking IndexUpdateResult."""
    return type(
        "FakeIdx",
        (),
        {
            "chunks": chunks or [],
            "changed_sources": changed or [],
            "deleted_chunk_ids": deleted or [],
            "full_rebuild": full_rebuild,
        },
    )


# --------------------------------------------------------------------------- #
# default_options / env defaults
# --------------------------------------------------------------------------- #

class TestDefaultOptions:
    def test_defaults_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in rebuild.ENV_VARS:
            monkeypatch.delenv(key, raising=False)
        opts = rebuild.default_options()
        assert opts["data_dir"] == "./data"
        assert opts["cache_dir"] == "./.rag_cache"
        assert opts["persist_dir"] == "./vectorstore"
        assert opts["chunk_size"] == 900
        assert opts["chunk_overlap"] == 120
        assert opts["batch_size"] == 32

    def test_env_overrides_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_DATA_DIR", "/custom/data")
        monkeypatch.setenv("RAG_CACHE_DIR", "/custom/cache")
        monkeypatch.setenv("RAG_VECTORSTORE_DIR", "/custom/vs")
        monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
        monkeypatch.setenv("RAG_CHUNK_OVERLAP", "50")
        monkeypatch.setenv("RAG_VECTORSTORE_BATCH_SIZE", "64")
        opts = rebuild.default_options()
        assert opts["data_dir"] == "/custom/data"
        assert opts["cache_dir"] == "/custom/cache"
        assert opts["persist_dir"] == "/custom/vs"
        assert opts["chunk_size"] == 500
        assert opts["chunk_overlap"] == 50
        assert opts["batch_size"] == 64


# --------------------------------------------------------------------------- #
# build_parser
# --------------------------------------------------------------------------- #

class TestBuildParser:
    def test_no_args_defaults_to_full_mode_flags_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in rebuild.ENV_VARS:
            monkeypatch.delenv(key, raising=False)
        args = _parse()
        assert args.full_rebuild is False
        assert args.incremental is False
        assert args.data_dir == "./data"
        assert args.batch_size == 32
        assert args.chunk_size == 900
        assert args.chunk_overlap == 120

    def test_cli_overrides_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_VECTORSTORE_BATCH_SIZE", "64")
        args = _parse("--batch-size", "16", "--data-dir", "/cli/data")
        assert args.batch_size == 16
        assert args.data_dir == "/cli/data"

    def test_full_rebuild_flag(self) -> None:
        args = _parse("--full-rebuild")
        assert args.full_rebuild is True
        assert args.incremental is False

    def test_incremental_flag(self) -> None:
        args = _parse("--incremental")
        assert args.incremental is True
        assert args.full_rebuild is False

    def test_mutually_exclusive_flags_rejected(self) -> None:
        with pytest.raises(SystemExit):
            _parse("--full-rebuild", "--incremental")

    def test_unknown_flag_rejected(self) -> None:
        with pytest.raises(SystemExit):
            _parse("--bogus")


# --------------------------------------------------------------------------- #
# resolve_mode
# --------------------------------------------------------------------------- #

class TestResolveMode:
    def test_incremental_flag_returns_incremental(self) -> None:
        args = _parse("--incremental")
        assert rebuild.resolve_mode(args) == "incremental"

    def test_full_rebuild_flag_returns_full(self) -> None:
        args = _parse("--full-rebuild")
        assert rebuild.resolve_mode(args) == "full"

    def test_no_flag_defaults_to_full(self) -> None:
        """Historical behaviour: default when neither flag is given is full rebuild."""
        args = _parse()
        assert rebuild.resolve_mode(args) == "full"


# --------------------------------------------------------------------------- #
# run() — sync wiring
# --------------------------------------------------------------------------- #

class TestRunSyncWiring:
    """Verify that run() wires the correct arguments to sync_vectorstore
    in full vs incremental mode, using mocks for heavy dependencies."""

    @patch("src.retriever.sync_vectorstore")
    @patch("src.index_cache.load_or_update_chunks")
    def test_full_mode_forces_full_rebuild_and_empty_changes(
        self, mock_load, mock_sync
    ) -> None:
        mock_load.return_value = _fake_idx_result(
            chunks=["doc1", "doc2"],
            changed=["a.pdf"],
            deleted=["old_id"],
            full_rebuild=False,
        )
        args = _parse("--full-rebuild", "--batch-size", "16")
        result = rebuild.run(args)

        # force_rebuild=True passed to load_or_update_chunks in full mode
        mock_load.assert_called_once()
        assert mock_load.call_args.kwargs["force_rebuild"] is True

        # sync_vectorstore receives full_rebuild=True, empty changes
        mock_sync.assert_called_once()
        kw = mock_sync.call_args.kwargs
        assert kw["full_rebuild"] is True
        assert kw["changed_sources"] == []
        assert kw["deleted_chunk_ids"] == []
        assert kw["batch_size"] == 16

        assert result["mode"] == "full"
        assert result["chunk_count"] == 2
        assert result["deleted_count"] == 0

    @patch("src.retriever.sync_vectorstore")
    @patch("src.index_cache.load_or_update_chunks")
    def test_incremental_mode_passes_cache_changes(
        self, mock_load, mock_sync
    ) -> None:
        mock_load.return_value = _fake_idx_result(
            chunks=["doc1", "doc2", "doc3"],
            changed=["b.pdf", "c.pdf"],
            deleted=["stale_id1", "stale_id2"],
            full_rebuild=False,
        )
        args = _parse("--incremental", "--batch-size", "8")
        result = rebuild.run(args)

        # force_rebuild=False in incremental mode
        mock_load.assert_called_once()
        assert mock_load.call_args.kwargs["force_rebuild"] is False

        # sync_vectorstore receives the cache's changed/deleted/full_rebuild
        mock_sync.assert_called_once()
        kw = mock_sync.call_args.kwargs
        assert kw["full_rebuild"] is False
        assert kw["changed_sources"] == ["b.pdf", "c.pdf"]
        assert kw["deleted_chunk_ids"] == ["stale_id1", "stale_id2"]
        assert kw["batch_size"] == 8

        assert result["mode"] == "incremental"
        assert result["chunk_count"] == 3
        assert result["changed_count"] == 2
        assert result["deleted_count"] == 2

    @patch("src.retriever.sync_vectorstore")
    @patch("src.index_cache.load_or_update_chunks")
    def test_incremental_mode_with_cache_full_rebuild_flag(
        self, mock_load, mock_sync
    ) -> None:
        """When the cache itself reports full_rebuild=True, sync should
        receive full_rebuild=True even in incremental CLI mode."""
        mock_load.return_value = _fake_idx_result(
            chunks=["doc1"],
            changed=[],
            deleted=[],
            full_rebuild=True,
        )
        args = _parse("--incremental")
        result = rebuild.run(args)

        kw = mock_sync.call_args.kwargs
        assert kw["full_rebuild"] is True
        assert result["mode"] == "incremental"
        assert result["changed_count"] == 0

    @patch("src.retriever.sync_vectorstore")
    @patch("src.index_cache.load_or_update_chunks")
    def test_default_mode_is_full(self, mock_load, mock_sync) -> None:
        """No mode flag given → full rebuild."""
        mock_load.return_value = _fake_idx_result(chunks=["d1"])
        args = _parse()
        result = rebuild.run(args)
        assert result["mode"] == "full"
        assert mock_load.call_args.kwargs["force_rebuild"] is True
        assert mock_sync.call_args.kwargs["full_rebuild"] is True

    @patch("src.retriever.sync_vectorstore")
    @patch("src.index_cache.load_or_update_chunks")
    def test_invalid_batch_size_raises_before_sync(
        self, mock_load, mock_sync
    ) -> None:
        args = _parse("--batch-size", "0")
        with pytest.raises(ValueError, match="greater than 0"):
            rebuild.run(args)
        mock_sync.assert_not_called()


# --------------------------------------------------------------------------- #
# print_summary
# --------------------------------------------------------------------------- #

class TestPrintSummary:
    def test_outputs_all_fields(self, capsys: pytest.CaptureFixture) -> None:
        rebuild.print_summary("incremental", 100, 3, 2, 32, 5.5)
        out = capsys.readouterr().out
        assert "mode=incremental" in out
        assert "chunks=100" in out
        assert "changed=3" in out
        assert "deleted=2" in out
        assert "batch_size=32" in out
        assert "elapsed=5.5s" in out
