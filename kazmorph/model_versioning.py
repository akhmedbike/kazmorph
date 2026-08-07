"""
Версионирование CRF-моделей для retrain-цикла Active Learning.

Зачем: основной train_crf.py перетирает один models/crf_model.joblib. Для
retrain-loop нужно сохранять историю версий и уметь откатываться, если
новая модель хуже предыдущей на gold-set (защита от model collapse).

Схема:
  models/versions/crf_v{N}.joblib       — веса версии N
  models/versions/crf_v{N}.meta.json    — метаданные (гиперпараметры, дата,
                                          источник данных, gold-метрики)
  models/crf_current.json               — pointer на активную версию:
      {"version": "v3", "path": "models/versions/crf_v3.joblib",
       "promoted_at": "...", "baseline_pos_acc": 0.8902}
  models/crf_model.joblib               — legacy-путь (fallback для main.py,
                                          если pointer'а нет)

Версия N = (максимальный существующий номер) + 1. Промоут: новая модель
становится current только если проходит гейт (accuracy не упала).
Откат: current pointer возвращается на предыдущую версию.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

VERSIONS_DIR = Path("models/versions")
POINTER_PATH = Path("models/crf_current.json")
LEGACY_MODEL_PATH = Path("models/crf_model.joblib")

_VERSION_RE = re.compile(r"crf_v(\d+)\.joblib$")


# ============================================================
#  Чтение версий
# ============================================================


def list_versions(versions_dir: Path | str = VERSIONS_DIR) -> list[int]:
    """Вернуть номера существующих версий (отсортированные)."""
    vdir = Path(versions_dir)
    if not vdir.exists():
        return []
    nums = []
    for path in vdir.glob("crf_v*.joblib"):
        m = _VERSION_RE.search(path.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def next_version_number(versions_dir: Path | str = VERSIONS_DIR) -> int:
    """Следующий номер версии (макс+1, или 1 если версий нет)."""
    existing = list_versions(versions_dir)
    return (existing[-1] + 1) if existing else 1


def version_path(n: int, versions_dir: Path | str = VERSIONS_DIR) -> Path:
    """Путь к joblib-файлу версии n."""
    return Path(versions_dir) / f"crf_v{n}.joblib"


def version_meta_path(n: int, versions_dir: Path | str = VERSIONS_DIR) -> Path:
    """Путь к meta.json версии n."""
    return Path(versions_dir) / f"crf_v{n}.meta.json"


# ============================================================
#  Pointer на текущую версию
# ============================================================


def read_pointer(pointer_path: Path | str = POINTER_PATH) -> dict | None:
    """Прочитать crf_current.json. None если файла нет."""
    p = Path(pointer_path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_pointer(version: int, *, baseline_pos_acc: float | None = None,
                  extra: dict | None = None,
                  pointer_path: Path | str = POINTER_PATH,
                  versions_dir: Path | str = VERSIONS_DIR) -> dict:
    """Записать pointer на версию. Атомарная запись."""
    pointer = {
        "version": f"v{version}",
        "version_number": version,
        "path": str(version_path(version, versions_dir)),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    if baseline_pos_acc is not None:
        pointer["baseline_pos_acc"] = baseline_pos_acc
    if extra:
        pointer.update(extra)
    p = Path(pointer_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pointer, f, ensure_ascii=False, indent=2)
    tmp.replace(p)
    return pointer


def resolve_current_model_path(pointer_path: Path | str = POINTER_PATH,
                               legacy_path: Path | str = LEGACY_MODEL_PATH) -> str | None:
    """Вернуть путь к активной модели для загрузки в Pipeline.

    Приоритет:
      1. crf_current.json pointer (если есть и файл существует)
      2. legacy models/crf_model.joblib (обратная совместимость)
      3. None (модель не загружена)
    """
    pointer = read_pointer(pointer_path)
    if pointer and pointer.get("path"):
        if Path(pointer["path"]).exists():
            return pointer["path"]
    legacy = Path(legacy_path)
    if legacy.exists():
        return str(legacy)
    return None


# ============================================================
#  Сохранение и метаданные
# ============================================================


def save_version(
    source_joblib: Path | str,
    *,
    meta: dict,
    versions_dir: Path | str = VERSIONS_DIR,
) -> tuple[int, Path]:
    """Скопировать обученную модель в versions/ под новым номером.

    source_joblib — куда train_crf.py сохранил новую модель.
    meta          — словарь метаданных для crf_v{N}.meta.json.
    Возвращает (номер_версии, путь_к_joblib).
    """
    vdir = Path(versions_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    n = next_version_number(vdir)
    dst = version_path(n, vdir)
    shutil.copy2(str(source_joblib), str(dst))

    meta_full = {
        "version": f"v{n}",
        "version_number": n,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    meta_dst = version_meta_path(n, vdir)
    with open(meta_dst, "w", encoding="utf-8") as f:
        json.dump(meta_full, f, ensure_ascii=False, indent=2)
    return n, dst


def read_meta(n: int, versions_dir: Path | str = VERSIONS_DIR) -> dict | None:
    """Прочитать meta.json версии n."""
    p = version_meta_path(n, versions_dir)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ============================================================
#  Промоут / откат
# ============================================================


def promote(version: int, *, baseline_pos_acc: float | None = None,
            extra: dict | None = None,
            pointer_path: Path | str = POINTER_PATH,
            versions_dir: Path | str = VERSIONS_DIR) -> dict:
    """Сделать версию n текущей (записать pointer)."""
    if not version_path(version, versions_dir).exists():
        raise FileNotFoundError(f"Версия v{version} не найдена в {versions_dir}")
    return write_pointer(version, baseline_pos_acc=baseline_pos_acc,
                         extra=extra, pointer_path=pointer_path,
                         versions_dir=versions_dir)


def rollback(pointer_path: Path | str = POINTER_PATH,
             versions_dir: Path | str = VERSIONS_DIR) -> dict | None:
    """Откатиться на предыдущую версию. Возвращает новый pointer или None.

    Если текущая v3 → откат на v2. Если v1 — откатываться некуда, None.
    """
    pointer = read_pointer(pointer_path)
    if not pointer:
        return None
    current = pointer.get("version_number", 0)
    if current <= 1:
        return None
    prev = current - 1
    if not version_path(prev, versions_dir).exists():
        return None
    return promote(prev, pointer_path=pointer_path, versions_dir=versions_dir)


def mark_rejected(version: int, *, reason: str,
                  versions_dir: Path | str = VERSIONS_DIR) -> Path:
    """Переименовать crf_v{N}.joblib → crf_v{N}_REJECTED.joblib.

    Модель не прошла гейт accuracy — помечаем, чтобы не путать с валидными.
    Pointer на неё НЕ ставится (остаётся предыдущая версия).
    """
    src = version_path(version, versions_dir)
    if not src.exists():
        raise FileNotFoundError(f"Версия v{version} не найдена: {src}")
    dst = src.with_name(f"crf_v{version}_REJECTED.joblib")
    src.rename(dst)
    # Дописываем причину в meta.
    meta = read_meta(version, versions_dir) or {}
    meta["rejected"] = True
    meta["rejected_reason"] = reason
    meta["rejected_at"] = datetime.now(timezone.utc).isoformat()
    meta_path = version_meta_path(version, versions_dir)
    if meta_path.exists():
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return dst
