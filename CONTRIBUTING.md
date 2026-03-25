# Contributing to insurance-whittaker

Whittaker-Henderson smoothing is a mature method, but making it work well on real UK insurance data requires more care than the textbook version. Contributions that improve the practical usability are welcome.

## Reporting bugs

Open a GitHub Issue. Include:

- The Python and library version (`import insurance_whittaker; print(insurance_whittaker.__version__)`)
- A minimal reproducible example with the data shape you are working with (1D age curve, 2D joint graduation, etc.)
- What you expected to happen and what actually happened

Edge cases that come up: very sparse data in some cells, integer vs. float exposures, and cases where the lambda tuning produces degenerate results (over-smoothed to a flat line). All worth reporting.

## Requesting features

Open a GitHub Issue with the label `enhancement`. Describe the smoothing problem — the rating dimension, approximate data volumes, and what the current tool cannot do. Cross-classified graduation (3D or higher) is a known gap.

## Development setup

```bash
git clone https://github.com/burning-cost/insurance-whittaker.git
cd insurance-whittaker
uv sync --dev
uv run pytest
```

The library uses `uv` for dependency management. Python 3.10+ is required.

## Code style

- Type hints on all public functions and methods
- UK English in docstrings and documentation
- Docstrings follow NumPy format
- The core mathematical operations use NumPy and SciPy sparse matrices — keep numerical dependencies minimal
- Tests should cover both 1D and 2D graduation cases, and include at least one test with deliberately sparse data to catch edge cases

---

For questions or to discuss ideas before opening an issue, start a [Discussion](https://github.com/burning-cost/insurance-whittaker/discussions).
