# Copilot code review instructions — AMS

AMS (PyPI: `ltbams`) is CURENT LTB's scheduling-modeling framework, GPL-3.0. Package root: `ams/`. AMS depends on ANDES; the AMS↔ANDES boundary is `ams/interface.py`.

Focus reviews on the rules below. Skip anything not listed.

## Style & lint

- No emojis in source, comments, or docstrings.
- No multi-paragraph docstrings or comment blocks on obvious code; comments are for non-obvious "why" only.
- New `.py` files carry the GPL-3.0 header used elsewhere in `ams/`.

## ANDES coupling (active decoupling refactor)

AMS is mid-decoupling from ANDES internals. Flag these as regressions:

- New `from andes.<submodule> import ...` or `import andes.<submodule>`. Baseline is **45** such imports in `ams/`; it must not grow. Top-level / public `andes` API is fine.
- Re-introducing ANDES inheritance, e.g. `class System(andes.system.System)`. AMS switched to composition in Step 1.2.
- New uses of ANDES v2.0.0 deprecated APIs (all replaced in AMS already):
  - `andes.shared.coloredlogs` — removed; use stdlib `logging`.
  - `andes.shared.NCPUS_PHYSICAL` — renamed; use `NCPUS`.
  - `cache.df_in` / `cache.refresh('df_in')` — use `as_df(vin=True)`.
  - `set(src='u', ..., attr='v')` for online-status writes — use `set_status(idx, value)`.
  - Positional `set(src, idx, value, 'v')` — use keyword `attr='v'`.

## `e_str` expression notation

In `ams/routines/**/*.py`, within `e_str` strings:

- `@` = matrix multiplication.
- `mul(a, b)` = element-wise multiplication.
- `*` = scalar multiplication.
- `dot` = alias for scalar `*` (not `mul`).
- No catch-all `word*word` → `@` rewrite exists. Don't rely on one.

Flag any `e_str` using `*` between matrices/vectors where matmul is intended — it silently does element-wise or errors.

## Structural rules

- `ams/extension/eva.py` is third-party EV-aggregator tooling. Do not move it to `ams/models/`.

## Tests

- CI runs `pytest` on Ubuntu / macOS / Windows × Python 3.11 / 3.12 / 3.13. Don't re-flag failures CI already reports.
- For PRs touching `ams/interface.py`, `ams/system.py`, or `ams/routines/`, flag missing coverage for the changed behavior.

## PR conventions

- Branch name: `refactor/step-X.Y-short-description` for decoupling work; otherwise descriptive kebab-case.
- One logical step per PR. If a PR mixes unrelated changes, ask to split.
- PR body must fill `Fixes #<issue>` and concrete bullets under "Proposed Changes". Flag empty templates.

## Out of scope — do not comment on

- Architectural direction or roadmap decisions — owned by the maintainer.
- Cosmetic preferences not listed above.
- `docs/` phrasing unless factually wrong.

When unsure whether a rule applies, prefer silence over a speculative comment.
