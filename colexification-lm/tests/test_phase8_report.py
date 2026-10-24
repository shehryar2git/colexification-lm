"""Phase 8 tests: the report builder (the robustness module is tested in test_phase8_robustness.py)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.report import build_report as br


# ----------------------------------------------------------------- report --
def test_formatters():
    assert br.num(0.12345, 3) == "0.123" and br.num(np.nan) == "n/a"
    assert br.pval(0.0004) == "*p* < .001" and br.pval(0.0648) == "*p* = .065"
    assert br.ci(0.1, 0.2) == "[0.10, 0.20]" and br.pct(0.9787, 1) == "97.9%"
    assert br.thousands(54285) == "54,285"


def test_md_table_round_trips():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
    t = br.md_table(df)
    lines = t.splitlines()
    assert lines[0] == "| a | b |" and len(lines) == 4
    assert lines[3] == "| 2 |  |"                              # NaN renders empty, not "nan"


def test_template_substitution_and_missing_key():
    V = br.Values()
    V.add("x", "7", "results/fake.csv")
    V.add("unused", "9", "results/fake.csv")
    out, used, missing = br.substitute("value {{x}} and {{ missing }}", V)
    assert out == "value 7 and {{ missing }}"
    assert used == {"x"} and missing == {"missing"}


def test_report_is_built_and_every_number_has_a_source():
    """The built report must contain no leftover placeholder and must carry a provenance
    entry for every value it used."""
    import config
    out = config.REPORT_DIR / "report.md"
    vals = config.RESULTS_DIR / "report_values.json"
    if not out.exists() or not vals.exists():
        pytest.skip("report not built yet (run: python -m src.report.build_report)")
    text = out.read_text(encoding="utf-8")
    assert "{{" not in text, "unsubstituted placeholder in report.md"
    v = json.loads(vals.read_text())
    assert v["n_used"] >= 80
    assert all(x["source"] for x in v["values"].values())
    words = v["word_count_body_excluding_tables_and_references"]
    assert 4000 <= words <= 6000, f"report body is {words} words (target 4,000-6,000)"


def test_appendices_exist_after_build():
    import config
    d = config.REPORT_DIR / "appendix"
    if not (config.REPORT_DIR / "report.md").exists():
        pytest.skip("report not built yet")
    for name in ("A_concepts.md", "B_english_audit.md", "C_robustness.md", "D_figures.md", "E_pending.md"):
        assert (d / name).exists(), f"missing appendix {name}"
