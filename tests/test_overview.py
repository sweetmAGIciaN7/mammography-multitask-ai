"""Phase 6: the overview numbers are read from the committed results and match the README."""
import json

from mammo.experiments import overview as O


def test_overview_rows_and_numbers(tmp_path):
    assert O.main(["--out", str(tmp_path), "--nboot", "200"]) == 0
    s = json.load(open(tmp_path / "summary.json"))
    rows = s["overview_rows"]
    assert [r["group"] for r in rows] == ["leaky", "leaky", "partial", "honest", "honest", "honest"]
    assert rows[0]["auc"] == 0.962 and rows[1]["auc"] == 1.0
    assert round(rows[3]["auc"], 3) == 0.755 and round(rows[4]["auc"], 3) == 0.782 and round(rows[5]["auc"], 3) == 0.746
    lo, hi = s["cbis_official"]["auc_ci95"]
    assert lo < 0.746 < hi and hi - lo < 0.2
    assert (tmp_path / "overview_chart.png").exists() and (tmp_path / "model_diagram.png").exists()
