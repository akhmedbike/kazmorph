"""
Тесты харвестера (scripts/harvest_sources.py).

Сетевые вызовы НЕ делаются — RSS/HTML подменяются. Фокус на детерминированной
логике: дедупликация по хэшу, cleanup по expires_at/возрасту, HTML-очистка,
группировка по локальной дате.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import harvest_sources as hv


# ============================================================
#  text_hash — детерминизм и нормализация
# ============================================================


class TestTextHash:
    def test_deterministic(self):
        assert hv.text_hash("сәлем әлем") == hv.text_hash("сәлем әлем")

    def test_whitespace_normalized(self):
        # Лишние пробелы/переносы не влияют на хэш.
        assert hv.text_hash("сәлем   әлем") == hv.text_hash("сәлем әлем")
        assert hv.text_hash("сәлем\nәлем") == hv.text_hash("сәлем әлем")

    def test_case_normalized(self):
        assert hv.text_hash("СӘлем") == hv.text_hash("сәлем")

    def test_different_text_different_hash(self):
        assert hv.text_hash("aaa") != hv.text_hash("bbb")

    def test_hex_length(self):
        h = hv.text_hash("test")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ============================================================
#  _strip_html — очистка RSS summary
# ============================================================


class TestStripHtml:
    def test_removes_tags(self):
        assert hv._strip_html("<p>Привет</p>") == "Привет"

    def test_preserves_text_with_links(self):
        out = hv._strip_html('Читайте <a href="x">здесь</a> подробности')
        assert out == "Читайте здесь подробности"

    def test_entities_decoded(self):
        assert hv._strip_html("a&nbsp;b&amp;c") == "a b&c"

    def test_br_becomes_space(self):
        assert hv._strip_html("строка1<br/>строка2") == "строка1 строка2"

    def test_empty_input(self):
        assert hv._strip_html("") == ""

    def test_kazakh_text_preserved(self):
        html = "<p>Қазақстан Республикасының Конституциясы</p>"
        assert hv._strip_html(html) == "Қазақстан Республикасының Конституциясы"


# ============================================================
#  Дедупликация: load/save seen_hashes
# ============================================================


class TestSeenHashes:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "SEEN_HASHES_PATH", tmp_path / ".seen.json")
        hv.save_seen_hashes({"abc123", "def456"})
        loaded = hv.load_seen_hashes()
        assert loaded == {"abc123", "def456"}

    def test_load_missing_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "SEEN_HASHES_PATH", tmp_path / "nonexistent.json")
        assert hv.load_seen_hashes() == set()

    def test_load_corrupted_returns_empty(self, tmp_path, monkeypatch):
        p = tmp_path / ".seen.json"
        p.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(hv, "SEEN_HASHES_PATH", p)
        assert hv.load_seen_hashes() == set()

    def test_save_preserves_updated_at(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "SEEN_HASHES_PATH", tmp_path / ".seen.json")
        hv.save_seen_hashes({"x"})
        data = json.loads((tmp_path / ".seen.json").read_text(encoding="utf-8"))
        assert "updated_at" in data
        assert data["hashes"] == ["x"]


# ============================================================
#  save_article — структура файла, expires_at
# ============================================================


class TestSaveArticle:
    def test_writes_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        article = {"title": "T", "url": "http://x", "text": "Сәлем әлем", "published": "2026-01-01"}
        path = hv.save_article(article, "test_src", "2026-01-01", retention_days=7)
        assert path is not None and path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["source"] == "test_src"
        assert data["text"] == "Сәлем әлем"
        assert data["url"] == "http://x"
        assert data["retention_days"] == 7

    def test_expires_at_is_fetch_plus_retention(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        article = {"title": "T", "url": "u", "text": "текст для хэша", "published": ""}
        path = hv.save_article(article, "src", "2026-01-01", retention_days=5)
        data = json.loads(path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(data["fetched_at"])
        expires = datetime.fromisoformat(data["expires_at"])
        delta = expires - fetched
        assert abs(delta - timedelta(days=5)) < timedelta(seconds=5)

    def test_filename_contains_hash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        article = {"title": "T", "url": "u", "text": "уникальный текст", "published": ""}
        path = hv.save_article(article, "src", "2026-01-01", retention_days=7)
        expected_hash = hv.text_hash("уникальный текст")
        assert expected_hash in path.name


# ============================================================
#  cleanup_old_files — удаление по expires_at и возрасту
# ============================================================


class TestCleanup:
    def test_removes_expired_by_expires_at(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        d = tmp_path / "raw" / "2026-01-01"
        d.mkdir(parents=True)
        old_path = d / "old.json"
        # expires_at в прошлом.
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        old_path.write_text(json.dumps({"text": "x", "expires_at": past}), encoding="utf-8")
        removed = hv.cleanup_old_files(retention_days=7)
        assert removed == 1
        assert not old_path.exists()

    def test_keeps_fresh_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        d = tmp_path / "raw" / "2026-08-04"
        d.mkdir(parents=True)
        fresh = d / "fresh.json"
        future = (datetime.now(timezone.utc) + timedelta(days=6)).isoformat()
        fresh.write_text(json.dumps({"text": "x", "expires_at": future}), encoding="utf-8")
        removed = hv.cleanup_old_files(retention_days=7)
        assert removed == 0
        assert fresh.exists()

    def test_removes_by_file_age_when_no_expires(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        d = tmp_path / "raw" / "old"
        d.mkdir(parents=True)
        old = d / "ancient.json"
        old.write_text(json.dumps({"text": "no expires field"}), encoding="utf-8")
        # Делаем файл старым через os.utime (10 дней назад).
        import os
        old_time = datetime.now().timestamp() - 10 * 86400
        os.utime(old, (old_time, old_time))
        removed = hv.cleanup_old_files(retention_days=7)
        assert removed == 1
        assert not old.exists()

    def test_skips_dotfiles(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        d = tmp_path / "raw"
        d.mkdir(parents=True)
        dotfile = d / ".seen_hashes.json"
        dotfile.write_text("{}", encoding="utf-8")
        removed = hv.cleanup_old_files(retention_days=7)
        assert removed == 0
        assert dotfile.exists()

    def test_no_output_dir_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "nonexistent")
        assert hv.cleanup_old_files(retention_days=7) == 0


# ============================================================
#  local_today — таймзона Казахстана
# ============================================================


class TestLocalToday:
    def test_format(self):
        today = hv.local_today()
        assert len(today) == 10
        assert today.count("-") == 2

    def test_is_kz_date_not_utc(self):
        # В 20:00–24:00 UTC локальная дата уже «завтра» по UTC+5.
        # Эта проверка гарантирует, что мы используем +05, а не UTC.
        from datetime import datetime
        kz_now = datetime.now(hv.KZ_TZ)
        assert hv.local_today() == kz_now.strftime("%Y-%m-%d")
        assert hv.KZ_TZ.utcoffset(None) == timedelta(hours=5)


# ============================================================
#  load_sources — конфиг
# ============================================================


class TestLoadSources:
    def test_reads_real_config(self):
        # Реальный config/sources.yaml из репо.
        sources = hv.load_sources()
        assert len(sources) >= 1
        names = [s["name"] for s in sources]
        assert "inform_kz" in names
        inform = [s for s in sources if s["name"] == "inform_kz"][0]
        assert inform["type"] == "rss"
        assert inform["lang"] == "kk"
        assert "kaz.inform.kz" in inform["url"]

    def test_empty_config_raises(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("sources: []", encoding="utf-8")
        with pytest.raises(ValueError, match="нет источников"):
            hv.load_sources(str(cfg))


# ============================================================
#  run_harvest с замоканным fetch — end-to-end логика без сети
# ============================================================


class TestRunHarvestMocked:
    """Замокать _fetch_url → вернуть синтетический RSS XML, проверить дедуп."""

    SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Test</title>
  <item>
    <title>Новость 1</title>
    <link>http://example.com/1</link>
    <pubDate>Mon, 04 Aug 2026 10:00:00 +0500</pubDate>
    <description><![CDATA[<p>Это полный текст первой новости достаточной длины для прохождения фильтра в двести символов и более чтобы пайплайн получил реальный материал для морфологического анализа казахского языка. Добавляем ещё текст чтобы гарантированно превысить порог в двести символов после очистки от HTML тегов.</p>]]></description>
  </item>
  <item>
    <title>Новость 2</title>
    <link>http://example.com/2</link>
    <description>коротко</description>
  </item>
</channel></rss>"""

    def _make_single_source_config(self, tmp_path):
        """Временный config/sources.yaml с одним синтетическим источником.

        Реальный config/sources.yaml теперь содержит несколько источников
        (inform_kz, habar_24kz), и тесты, завязанные на точные счётчики,
        ломаются при их добавлении. Изоляция через свой конфиг решает это.
        """
        import yaml
        cfg_path = tmp_path / "sources.yaml"
        cfg = {
            "sources": [{
                "name": "test_src",
                "type": "rss",
                "url": "http://test.local/rss",
                "lang": "kk",
                "max_articles": 10,
                "min_text_length": 200,
            }]
        }
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return str(cfg_path)

    def test_harvest_dedup_and_filter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        monkeypatch.setattr(hv, "SEEN_HASHES_PATH", tmp_path / "raw" / ".seen.json")
        monkeypatch.setattr(hv, "_fetch_url", lambda url, **kw: self.SAMPLE_RSS.encode("utf-8"))
        monkeypatch.setattr(hv, "CONFIG_PATH", self._make_single_source_config(tmp_path))

        # Первый прогон: 1 годная статья (вторая отфильтрована по длине).
        stats1 = hv.run_harvest()
        assert stats1["total_new"] == 1
        assert stats1["total_skipped_dup"] == 0
        # Второй прогон: тот же fetch → дубликат.
        stats2 = hv.run_harvest()
        assert stats2["total_new"] == 0
        assert stats2["total_skipped_dup"] == 1

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hv, "OUTPUT_DIR", tmp_path / "raw")
        monkeypatch.setattr(hv, "SEEN_HASHES_PATH", tmp_path / "raw" / ".seen.json")
        monkeypatch.setattr(hv, "_fetch_url", lambda url, **kw: self.SAMPLE_RSS.encode("utf-8"))
        monkeypatch.setattr(hv, "CONFIG_PATH", self._make_single_source_config(tmp_path))
        stats = hv.run_harvest(dry_run=True)
        assert stats["total_new"] == 1
        assert stats["dry_run"] is True
        assert not (tmp_path / "raw").exists() or not list((tmp_path / "raw").rglob("*.json"))
