"""
Тесты версионирования CRF-моделей (kazmorph/model_versioning.py).

Изоляция через tmp_path — реальные models/ не трогаются. Фокус на логике
нумерации версий, pointer'е, промоуте/откате/REJECTED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kazmorph import model_versioning as mv


# ============================================================
#  list_versions / next_version_number
# ============================================================


class TestListVersions:
    def test_empty_dir(self, tmp_path):
        assert mv.list_versions(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert mv.list_versions(tmp_path / "nope") == []

    def test_finds_versions(self, tmp_path):
        for n in [1, 3, 2]:
            (tmp_path / f"crf_v{n}.joblib").write_bytes(b"x")
        assert mv.list_versions(tmp_path) == [1, 2, 3]

    def test_ignores_rejected(self, tmp_path):
        (tmp_path / "crf_v1.joblib").write_bytes(b"x")
        (tmp_path / "crf_v2_REJECTED.joblib").write_bytes(b"x")
        (tmp_path / "crf_v3.joblib").write_bytes(b"x")
        assert mv.list_versions(tmp_path) == [1, 3]

    def test_ignores_non_version_files(self, tmp_path):
        (tmp_path / "crf_v1.joblib").write_bytes(b"x")
        (tmp_path / "crf_model.joblib").write_bytes(b"x")  # legacy
        (tmp_path / "random.txt").write_text("x")
        assert mv.list_versions(tmp_path) == [1]


class TestNextVersion:
    def test_empty_returns_one(self, tmp_path):
        assert mv.next_version_number(tmp_path) == 1

    def test_after_max(self, tmp_path):
        for n in [1, 2, 5]:
            (tmp_path / f"crf_v{n}.joblib").write_bytes(b"x")
        assert mv.next_version_number(tmp_path) == 6


class TestVersionPaths:
    def test_version_path(self, tmp_path):
        assert mv.version_path(7, tmp_path) == tmp_path / "crf_v7.joblib"

    def test_meta_path(self, tmp_path):
        assert mv.version_meta_path(3, tmp_path) == tmp_path / "crf_v3.meta.json"


# ============================================================
#  Pointer (read/write)
# ============================================================


class TestPointer:
    def test_write_and_read(self, tmp_path):
        pointer_path = tmp_path / "crf_current.json"
        mv.write_pointer(3, baseline_pos_acc=0.89, pointer_path=pointer_path,
                         versions_dir=tmp_path)
        pointer = mv.read_pointer(pointer_path)
        assert pointer["version"] == "v3"
        assert pointer["version_number"] == 3
        assert pointer["baseline_pos_acc"] == 0.89
        assert str(tmp_path / "crf_v3.joblib") in pointer["path"]

    def test_read_missing_returns_none(self, tmp_path):
        assert mv.read_pointer(tmp_path / "nope.json") is None

    def test_read_corrupted_returns_none(self, tmp_path):
        p = tmp_path / "crf_current.json"
        p.write_text("{broken", encoding="utf-8")
        assert mv.read_pointer(p) is None

    def test_write_extra_fields(self, tmp_path):
        pointer_path = tmp_path / "crf_current.json"
        mv.write_pointer(2, extra={"source": "rss_llm", "n_corrections": 150},
                         pointer_path=pointer_path, versions_dir=tmp_path)
        pointer = mv.read_pointer(pointer_path)
        assert pointer["source"] == "rss_llm"
        assert pointer["n_corrections"] == 150


# ============================================================
#  resolve_current_model_path
# ============================================================


class TestResolveCurrent:
    def test_uses_pointer_if_file_exists(self, tmp_path):
        vdir = tmp_path / "versions"
        vdir.mkdir()
        (vdir / "crf_v2.joblib").write_bytes(b"weights")
        pointer_path = tmp_path / "crf_current.json"
        mv.write_pointer(2, pointer_path=pointer_path, versions_dir=vdir)
        resolved = mv.resolve_current_model_path(pointer_path=pointer_path,
                                                  legacy_path=tmp_path / "legacy.joblib")
        assert resolved is not None
        assert "crf_v2.joblib" in resolved

    def test_falls_back_to_legacy(self, tmp_path):
        legacy = tmp_path / "crf_model.joblib"
        legacy.write_bytes(b"legacy")
        resolved = mv.resolve_current_model_path(
            pointer_path=tmp_path / "nope.json", legacy_path=legacy)
        assert resolved == str(legacy)

    def test_returns_none_if_nothing_exists(self, tmp_path):
        resolved = mv.resolve_current_model_path(
            pointer_path=tmp_path / "nope.json", legacy_path=tmp_path / "nope2.joblib")
        assert resolved is None

    def test_pointer_to_missing_file_falls_back(self, tmp_path):
        # Pointer указывает на v3, но файла нет → fallback на legacy.
        vdir = tmp_path / "versions"
        vdir.mkdir()
        pointer_path = tmp_path / "crf_current.json"
        mv.write_pointer(3, pointer_path=pointer_path, versions_dir=vdir)
        legacy = tmp_path / "crf_model.joblib"
        legacy.write_bytes(b"legacy")
        resolved = mv.resolve_current_model_path(pointer_path=pointer_path,
                                                  legacy_path=legacy)
        assert resolved == str(legacy)


# ============================================================
#  save_version + meta
# ============================================================


class TestSaveVersion:
    def test_copies_and_assigns_number(self, tmp_path):
        source = tmp_path / "trained.joblib"
        source.write_bytes(b"new-weights")
        vdir = tmp_path / "versions"
        n, dst = mv.save_version(source, meta={"c1": 0.2, "c2": 0.01},
                                 versions_dir=vdir)
        assert n == 1
        assert dst == vdir / "crf_v1.joblib"
        assert dst.read_bytes() == b"new-weights"

    def test_meta_written(self, tmp_path):
        source = tmp_path / "trained.joblib"
        source.write_bytes(b"w")
        vdir = tmp_path / "versions"
        n, _ = mv.save_version(source, meta={"c1": 0.2, "baseline_pos_acc": 0.89},
                               versions_dir=vdir)
        meta = mv.read_meta(n, vdir)
        assert meta["version"] == "v1"
        assert meta["c1"] == 0.2
        assert meta["baseline_pos_acc"] == 0.89
        assert "saved_at" in meta

    def test_multiple_versions_increment(self, tmp_path):
        vdir = tmp_path / "versions"
        for i in range(3):
            src = tmp_path / f"s{i}.joblib"
            src.write_bytes(f"w{i}".encode())
            n, _ = mv.save_version(src, meta={"i": i}, versions_dir=vdir)
            assert n == i + 1
        assert mv.list_versions(vdir) == [1, 2, 3]

    def test_read_meta_missing(self, tmp_path):
        assert mv.read_meta(99, tmp_path) is None


# ============================================================
#  promote / rollback / mark_rejected
# ============================================================


class TestPromoteRollback:
    def _setup_three_versions(self, tmp_path):
        vdir = tmp_path / "versions"
        vdir.mkdir()
        for n in [1, 2, 3]:
            (vdir / f"crf_v{n}.joblib").write_bytes(f"w{n}".encode())
            with open(vdir / f"crf_v{n}.meta.json", "w") as f:
                json.dump({"version": f"v{n}"}, f)
        pointer_path = tmp_path / "crf_current.json"
        return vdir, pointer_path

    def test_promote_sets_pointer(self, tmp_path):
        vdir, pointer_path = self._setup_three_versions(tmp_path)
        mv.promote(2, pointer_path=pointer_path, versions_dir=vdir)
        pointer = mv.read_pointer(pointer_path)
        assert pointer["version_number"] == 2
        assert pointer["path"] == str(vdir / "crf_v2.joblib")

    def test_promote_missing_version_raises(self, tmp_path):
        vdir, pointer_path = self._setup_three_versions(tmp_path)
        with pytest.raises(FileNotFoundError):
            mv.promote(99, pointer_path=pointer_path, versions_dir=vdir)

    def test_rollback_to_previous(self, tmp_path):
        vdir, pointer_path = self._setup_three_versions(tmp_path)
        mv.promote(3, pointer_path=pointer_path, versions_dir=vdir)
        new_pointer = mv.rollback(pointer_path=pointer_path, versions_dir=vdir)
        assert new_pointer["version_number"] == 2

    def test_rollback_from_v1_returns_none(self, tmp_path):
        vdir = tmp_path / "versions"
        vdir.mkdir()
        (vdir / "crf_v1.joblib").write_bytes(b"w1")
        pointer_path = tmp_path / "crf_current.json"
        mv.promote(1, pointer_path=pointer_path, versions_dir=vdir)
        assert mv.rollback(pointer_path=pointer_path, versions_dir=vdir) is None

    def test_rollback_no_pointer_returns_none(self, tmp_path):
        vdir, pointer_path = self._setup_three_versions(tmp_path)
        assert mv.rollback(pointer_path=pointer_path, versions_dir=vdir) is None

    def test_mark_rejected_renames_and_flags(self, tmp_path):
        vdir = tmp_path / "versions"
        vdir.mkdir()
        (vdir / "crf_v2.joblib").write_bytes(b"w2")
        with open(vdir / "crf_v2.meta.json", "w") as f:
            json.dump({"version": "v2"}, f)
        rejected_path = mv.mark_rejected(2, reason="pos_acc dropped 0.85→0.82",
                                         versions_dir=vdir)
        assert rejected_path.name == "crf_v2_REJECTED.joblib"
        assert rejected_path.exists()
        assert not (vdir / "crf_v2.joblib").exists()  # переименован
        # Не попадает в list_versions.
        assert 2 not in mv.list_versions(vdir)
        # meta помечена.
        meta = json.loads((vdir / "crf_v2.meta.json").read_text(encoding="utf-8"))
        assert meta["rejected"] is True
        assert "pos_acc dropped" in meta["rejected_reason"]
