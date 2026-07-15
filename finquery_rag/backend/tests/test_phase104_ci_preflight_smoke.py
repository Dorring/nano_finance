import importlib.util
import json
from pathlib import Path


def test_ci_preflight_smoke_script_writes_passing_report(tmp_path, monkeypatch):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "ci_preflight_smoke.py"
    spec = importlib.util.spec_from_file_location("ci_preflight_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    artifact_dir = tmp_path / "preflight-artifacts"
    monkeypatch.setenv("FINQUERY_PREFLIGHT_ARTIFACT_DIR", str(artifact_dir))

    code = module.main()

    report = json.loads((artifact_dir / "preflight_smoke.json").read_text(encoding="utf-8"))
    assert code == 0
    assert report["passed"] is True
    assert report["sections"] == {
        "health": True,
        "migration": True,
        "fixture_audit": True,
        "eval_gate": True,
        "baseline_comparison": True,
    }
    assert report["summary"]["retrieval_recall_at_5"] == 1.0
