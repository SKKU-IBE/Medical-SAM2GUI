# Contributing

Thank you for helping improve Interactive Medical-SAM2 GUI. Bug reports,
focused fixes, tests, and documentation improvements are welcome.

## Development setup

Use a fresh clone. The repository history was rewritten before v1.1.0 to
remove files that could not be redistributed; see [HISTORY_REWRITE.md](HISTORY_REWRITE.md).

```bash
git clone https://github.com/SKKU-IBE/Medical-SAM2GUI.git
cd Medical-SAM2GUI
uv sync --frozen
uv run pytest -q
```

The Medical-SAM2 checkpoint is not required for the automated test suite. For
interactive propagation, review the upstream terms and run:

```bash
uv run medical-sam2-download-checkpoint
```

## Pull requests

1. Open an issue first for broad behavior changes or changes to saved formats.
2. Keep changes scoped and add regression tests for user-visible behavior.
3. Run `uv sync --frozen`, `uv run pytest -q`, and `uv build`.
4. Describe manual GUI validation when a change affects Napari interaction.
5. Update README or changelog text when commands, outputs, or compatibility change.

Do not commit checkpoints, source medical-image volumes, patient screenshots,
institutional paths, identifiers, or exported masks derived from non-public
data. Public demonstration media must have documented redistribution rights,
case-level provenance, processing steps, and attribution in
[MEDIA_PROVENANCE.md](MEDIA_PROVENANCE.md).

## Bug reports

Include the operating system, Python version, input format, image geometry, and
the shortest reproducible sequence of actions. Redact personal, patient, and
institutional information from logs and paths. For security or privacy issues,
contact the maintainers privately instead of opening a public issue containing
sensitive data.

## Licensing

Contributions are accepted under `GPL-3.0-only`. Do not copy code, models, or
media whose license is incompatible or unclear. Preserve upstream notices for
adapted third-party code and update [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
when necessary.
