import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "audios_EMIS.zip"
DEST = ROOT / "audios_EMIS_balanced144.zip"
MANIFEST = ROOT / "emis_balanced144_manifest.jsonl"
ALIGNED_DEST = ROOT / "audios_EMIS_aligned48.zip"
ALIGNED_MANIFEST = ROOT / "emis_aligned48_manifest.jsonl"
PATTERN = re.compile(
    r"^(?P<text_id>\d+)_(?P<semantic>happy|sad|angry)_(?P<explicitness>explicit|implicit)_"
    r"(?P<acoustic>happy|sad|angry|neutral)_(?P<voice_id>\d+)_(?P<generator>F5TTS|COSY|STYLE)\.wav$|"
    r"^(?P<neutral_text_id>\d+)_neutral_(?P<neutral_acoustic>happy|sad|angry|neutral)_"
    r"(?P<neutral_voice_id>\d+)_(?P<neutral_generator>F5TTS|COSY|STYLE)\.wav$"
)


def parse(name):
    match = PATTERN.match(name)
    if not match:
        raise ValueError(name)
    data = match.groupdict()
    if data["neutral_text_id"] is not None:
        return {
            "filename": name,
            "text_id": int(data["neutral_text_id"]),
            "semantic_label": "neutral",
            "explicitness": "neutral",
            "acoustic_label": data["neutral_acoustic"],
            "voice_id": int(data["neutral_voice_id"]),
            "generator": data["neutral_generator"],
        }
    return {
        "filename": name,
        "text_id": int(data["text_id"]),
        "semantic_label": data["semantic"],
        "explicitness": data["explicitness"],
        "acoustic_label": data["acoustic"],
        "voice_id": int(data["voice_id"]),
        "generator": data["generator"],
    }


with zipfile.ZipFile(SOURCE) as source:
    rows = [parse(name) for name in source.namelist() if name.endswith(".wav")]
    conflicts = [row for row in rows if row["semantic_label"] != row["acoustic_label"]]
    buckets = defaultdict(list)
    for row in conflicts:
        buckets[(row["generator"], row["semantic_label"], row["acoustic_label"], row["explicitness"])].append(row)
    selected = []
    for generator in ("COSY", "F5TTS", "STYLE"):
        for semantic in ("neutral", "happy", "sad", "angry"):
            for acoustic in ("neutral", "happy", "sad", "angry"):
                if semantic == acoustic:
                    continue
                if semantic == "neutral":
                    candidates = sorted(buckets[(generator, semantic, acoustic, "neutral")], key=lambda r: (r["text_id"], r["voice_id"]))
                    selected.extend(candidates[:4])
                else:
                    for explicitness in ("explicit", "implicit"):
                        candidates = sorted(buckets[(generator, semantic, acoustic, explicitness)], key=lambda r: (r["text_id"], r["voice_id"]))
                        selected.extend(candidates[:2])
    assert len(selected) == 144, len(selected)
    counts = Counter((r["generator"], r["semantic_label"], r["acoustic_label"]) for r in selected)
    assert set(counts.values()) == {4}, counts
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as dest:
        for row in selected:
            dest.writestr(row["filename"], source.read(row["filename"]))

    aligned = []
    for generator in ("COSY", "F5TTS", "STYLE"):
        for semantic in ("neutral", "happy", "sad", "angry"):
            candidates = sorted(
                (row for row in rows if row["generator"] == generator and row["semantic_label"] == semantic and row["acoustic_label"] == semantic),
                key=lambda row: (row["explicitness"], row["text_id"], row["voice_id"]),
            )
            aligned.extend(candidates[:4])
    assert len(aligned) == 48, len(aligned)
    with zipfile.ZipFile(ALIGNED_DEST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as dest:
        for row in aligned:
            dest.writestr(row["filename"], source.read(row["filename"]))

with MANIFEST.open("w", encoding="utf-8") as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
with ALIGNED_MANIFEST.open("w", encoding="utf-8") as handle:
    for row in aligned:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"conflict": len(selected), "conflict_zip_bytes": DEST.stat().st_size, "cells": len(counts), "aligned": len(aligned), "aligned_zip_bytes": ALIGNED_DEST.stat().st_size}, ensure_ascii=False))
