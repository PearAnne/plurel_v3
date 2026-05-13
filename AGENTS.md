# Repository Guidelines

## Project Structure & Module Organization

`plurel/` is the installable Python package for synthetic relational database generation. `rt/` contains Relational Transformer data, embedding, model, task, and training code. `rustler/` is a separate Rust/PyO3 context sampler built with Maturin. `scripts/` holds experiment entry points such as `synthetic_gen.py`, `baseline_pretrain.py`, `synthetic_pretrain.py`, and `cntd_pretrain.py`. `test/` contains pytest coverage for the Python package. `docs/` and `docs/static/` provide the project site assets, while `examples/` contains notebooks such as SQL-schema synthesis examples.

## Build, Test, and Development Commands

Use Pixi for the development environment:

```bash
pixi install
cd rustler && pixi run maturin develop --uv --release && cd ..
pixi run pytest
pixi run ruff check .
pixi run ruff format .
```

`pixi install` creates the Python 3.12 environment. The Maturin command compiles and installs the Rust sampler. `pixi run pytest` runs the configured test suite. Ruff handles linting, import ordering, and formatting. For data generation, use `pixi run python scripts/synthetic_gen.py --seed_offset 0 --num_dbs 10 --num_proc 4`. For pretraining smoke runs, use `pixi run torchrun --standalone --nproc_per_node=1 scripts/synthetic_pretrain.py`.

## Coding Style & Naming Conventions

Python targets 3.12. Ruff is configured in `pyproject.toml` with line length 100, double quotes, and rules `E`, `F`, `I`, and `UP`. Use 4-space indentation, `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. New Python functions should include type annotations; prefer `pathlib.Path` for paths. Keep scripts as thin entry points and put reusable logic in `plurel/` or `rt/`. If editing `rustler/`, follow standard Rust formatting and keep the extension buildable through Maturin.

## Testing Guidelines

Tests live in `test/` and are discovered by pytest. Name files `test_<area>.py` and tests `test_<expected_behavior>()`. Prefer focused tests for changed behavior, then run the full suite:

```bash
pixi run pytest test/test_schema.py -q
pixi run pytest
```

The repository pytest config uses `pythonpath = .`, reports the 10 slowest tests, stops after the first failure, and runs with xdist workers.

## Commit & Pull Request Guidelines

Recent history uses short, direct commit subjects such as `fix relbench dependency`, `limit threads during parallel generation`, and `update README`. Keep commits focused on one logical change and use concise imperative or descriptive lower-case subjects. Pull requests should explain the motivation, list changed modules or scripts, link related issues when available, and include the exact commands run. Add screenshots only for `docs/` or visual asset changes. Do not commit generated caches, downloaded checkpoints, or large synthetic datasets.
