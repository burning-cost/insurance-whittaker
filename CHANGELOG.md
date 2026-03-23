# Changelog

## v0.1.2 (2026-03-22) [unreleased]
- Bump to 0.1.2 to publish Kronecker ordering fix to PyPI
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.1.1 (2026-03-21)
- docs: replace pip install with uv add in README
- Add blog post link and community CTA to README
- Add MIT license
- Fix README accuracy: remove false banded Cholesky complexity claim
- Add PyPI classifiers for financial/insurance audience
- Add Colab quickstart notebook for Whittaker-Henderson smoothing
- Add CONTRIBUTING.md with bug reporting, feature request, and dev setup guidance
- refresh benchmark numbers post-P0 fixes
- Fix P0/P1 bugs: Kronecker ordering, sigma^2 CIs, PIRLS lambda, 2D REML
- Fix docs workflow: use pdoc not pdoc3 syntax (no --html flag)
- Add pdoc API documentation workflow with GitHub Pages deployment
- Add benchmark: Whittaker-Henderson vs raw rates and weighted moving average
- fix: relax REML constant-data tolerance from 1e-5 to 1e-4
- Add shields.io badge row to README
- docs: add Databricks notebook link
- Add Related Libraries section to README

