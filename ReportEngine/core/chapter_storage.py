"""
Chapter JSON persistence and manifest management.

Each chapter is immediately written to a raw file during streaming generation.
After validation passes, it is written to a formatted chapter.json, with
metadata recorded in the manifest for subsequent stitching.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional


@dataclass
class ChapterRecord:
    """
    Chapter metadata recorded in the manifest.

    This structure is used to track each chapter's status, file location,
    and possible error list in manifest.json, making it easy for frontend
    or debugging tools to read.
    """

    chapter_id: str
    slug: str
    title: str
    order: int
    status: str
    files: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, object]:
        """Converts the record to a serializable dict suitable for writing to manifest.json."""
        return {
            "chapterId": self.chapter_id,
            "slug": self.slug,
            "title": self.title,
            "order": self.order,
            "status": self.status,
            "files": self.files,
            "errors": self.errors,
            "updatedAt": self.updated_at,
        }


class ChapterStorage:
    """
    Chapter JSON writer and manifest manager.

    Responsible for:
        - Creating an independent run directory and manifest snapshot for each report;
        - Immediately writing to stream.raw during chapter streaming generation;
        - Persisting chapter.json and updating manifest status after validation passes.
    """

    def __init__(self, base_dir: str):
        """
        Create chapter storage.

        Args:
            base_dir: Root path for all output run directories.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifests: Dict[str, Dict[str, object]] = {}

    # ======== Session & Manifest ========

    def start_session(self, report_id: str, metadata: Dict[str, object]) -> Path:
        """
        Create an independent chapter output directory and manifest for this report.

        Also writes global metadata to manifest.json for rendering/debugging queries.

        Args:
            report_id: Task ID.
            metadata: Report metadata (title, topic, etc.).

        Returns:
            Path: The newly created run directory.
        """
        run_dir = self.base_dir / report_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "reportId": report_id,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata,
            "chapters": [],
        }
        self._manifests[self._key(run_dir)] = manifest
        self._write_manifest(run_dir, manifest)
        return run_dir

    def begin_chapter(self, run_dir: Path, chapter_meta: Dict[str, object]) -> Path:
        """
        Create chapter subdirectory and mark it as streaming status in the manifest.

        Generates an order-slug style subdirectory and pre-registers the raw file path.

        Args:
            run_dir: Session root directory.
            chapter_meta: Metadata containing chapterId/title/slug/order.

        Returns:
            Path: The chapter directory.
        """
        slug_value = str(
            chapter_meta.get("slug") or chapter_meta.get("chapterId") or "section"
        )
        chapter_dir = self._chapter_dir(
            run_dir,
            slug_value,
            int(chapter_meta.get("order", 0)),
        )
        record = ChapterRecord(
            chapter_id=str(chapter_meta.get("chapterId")),
            slug=slug_value,
            title=str(chapter_meta.get("title")),
            order=int(chapter_meta.get("order", 0)),
            status="streaming",
            files={"raw": str(self._raw_stream_path(chapter_dir).relative_to(run_dir))},
        )
        self._upsert_record(run_dir, record)
        return chapter_dir

    def persist_chapter(
        self,
        run_dir: Path,
        chapter_meta: Dict[str, object],
        payload: Dict[str, object],
        errors: Optional[List[str]] = None,
    ) -> Path:
        """
        Write final JSON and update manifest status after chapter streaming is complete.

        If validation fails, error information is written to the manifest for frontend display.

        Args:
            run_dir: Session root directory.
            chapter_meta: Chapter metadata.
            payload: Validated chapter JSON.
            errors: Optional error list for marking invalid status.

        Returns:
            Path: The final chapter.json file path.
        """
        slug_value = str(
            chapter_meta.get("slug") or chapter_meta.get("chapterId") or "section"
        )
        chapter_dir = self._chapter_dir(
            run_dir,
            slug_value,
            int(chapter_meta.get("order", 0)),
        )
        final_path = chapter_dir / "chapter.json"
        final_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        record = ChapterRecord(
            chapter_id=str(chapter_meta.get("chapterId")),
            slug=slug_value,
            title=str(chapter_meta.get("title")),
            order=int(chapter_meta.get("order", 0)),
            status="ready" if not errors else "invalid",
            files={
                "raw": str(self._raw_stream_path(chapter_dir).relative_to(run_dir)),
                "json": str(final_path.relative_to(run_dir)),
            },
            errors=errors or [],
        )
        self._upsert_record(run_dir, record)
        return final_path

    def load_chapters(self, run_dir: Path) -> List[Dict[str, object]]:
        """
        Read all chapter.json files from the specified run directory and return sorted by order.

        Commonly used by DocumentComposer to stitch multiple chapters into a complete IR.

        Args:
            run_dir: Session root directory.

        Returns:
            list[dict]: List of chapter payloads.
        """
        payloads: List[Dict[str, object]] = []
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            chapter_path = child / "chapter.json"
            if not chapter_path.exists():
                continue
            try:
                payload = json.loads(chapter_path.read_text(encoding="utf-8"))
                payloads.append(payload)
            except json.JSONDecodeError:
                continue
        payloads.sort(key=lambda x: x.get("order", 0))
        return payloads

    # ======== File Operations ========

    @contextmanager
    def capture_stream(self, chapter_dir: Path) -> Generator:
        """
        Write streaming output to the raw file in real-time.

        Exposes a file handle via contextmanager to simplify chapter node write logic.

        Args:
            chapter_dir: Current chapter directory.

        Returns:
            Generator[TextIO]: File object used as a context manager.
        """
        raw_path = self._raw_stream_path(chapter_dir)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as fp:
            yield fp

    # ======== Internal Utilities ========

    def _chapter_dir(self, run_dir: Path, slug: str, order: int) -> Path:
        """Generate a stable directory based on slug/order, ensuring chapters are stored separately and sortable."""
        safe_slug = self._safe_slug(slug)
        folder = f"{order:03d}-{safe_slug}"
        path = run_dir / folder
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_slug(self, slug: str) -> str:
        """Remove dangerous characters to avoid generating illegal folder names."""
        slug = slug.replace(" ", "-").replace("/", "-")
        return slug or "section"

    def _raw_stream_path(self, chapter_dir: Path) -> Path:
        """Returns the raw file path corresponding to a chapter's streaming output."""
        return chapter_dir / "stream.raw"

    def _key(self, run_dir: Path) -> str:
        """Parse run directory into a dictionary cache key to avoid repeated disk reads."""
        return str(run_dir.resolve())

    def _manifest_path(self, run_dir: Path) -> Path:
        """Get the actual file path of manifest.json."""
        return run_dir / "manifest.json"

    def _write_manifest(self, run_dir: Path, manifest: Dict[str, object]):
        """Write the in-memory manifest snapshot fully back to disk."""
        self._manifest_path(run_dir).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_manifest(self, run_dir: Path) -> Dict[str, object]:
        """
        Read existing manifest from disk.

        Can be used to restore context when the process restarts or multiple instances write to disk.
        """
        manifest_path = self._manifest_path(run_dir)
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"reportId": run_dir.name, "chapters": []}

    def _upsert_record(self, run_dir: Path, record: ChapterRecord):
        """
        Update or append a chapter record in the manifest, ensuring consistent ordering.

        Automatically sorts and writes back to cache and disk.
        """
        key = self._key(run_dir)
        manifest = self._manifests.get(key) or self._read_manifest(run_dir)
        chapters: List[Dict[str, object]] = manifest.get("chapters", [])
        chapters = [c for c in chapters if c.get("chapterId") != record.chapter_id]
        chapters.append(record.to_dict())
        chapters.sort(key=lambda x: x.get("order", 0))
        manifest["chapters"] = chapters
        manifest.setdefault("updatedAt", datetime.utcnow().isoformat() + "Z")
        self._manifests[key] = manifest
        self._write_manifest(run_dir, manifest)


__all__ = ["ChapterStorage", "ChapterRecord"]
