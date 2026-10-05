# unslop audit — how to read this folder

Tool: https://github.com/theclaymethod/unslop (cloned 2026-10-05, stdlib scanners run as-is).
Per doc: `<name>_phrase.json` (banned_phrase_scan), `<name>_structure.json` (structure_scan),
`<name>_silhouette.json` (silhouette_scan); co-writer `suggest.py` returned zero suggestions
on all docs (not stored — nothing to store).

Verdict: phrase 0 violations everywhere (a few `non_english` declines are the language
detector choking on tables/symbols, not prose); structure 0 flags everywhere (short docs
decline cleanly); silhouette `heading_preview`/`callback_content` flags are genre-required
in technical reports (plan-mirroring headings, repeated model/dataset names) — restructuring
would damage clarity, so per the tool's own do-no-harm doctrine no prose was changed.
See commit message for the full reasoning.
