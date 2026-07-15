import importlib.util
import json
from pathlib import Path


def test_ci_eval_gate_script_writes_artifacts_to_env_dir(tmp_path, monkeypatch):
    # Resolve from this test file location so the sparse checkout can move.
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "ci_eval_gate.py"
    spec = importlib.util.spec_from_file_location("ci_eval_gate", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("FINQUERY_EVAL_ARTIFACT_DIR", str(artifact_dir))

    code = module.main()

    audit = json.loads((artifact_dir / "smoke_fixture_audit.json").read_text(encoding="utf-8"))
    report = json.loads((artifact_dir / "smoke_report.json").read_text(encoding="utf-8"))
    comparison = json.loads((artifact_dir / "smoke_comparison.json").read_text(encoding="utf-8"))
    junit = (artifact_dir / "smoke_gate.xml").read_text(encoding="utf-8")
    retrieval = json.loads((artifact_dir / "smoke_retrieval_diagnostics.json").read_text(encoding="utf-8"))
    assert code == 0
    assert audit["passed"] is True
    assert audit["summary"]["total_cases"] == 3
    assert report["summary"]["pass_rate"] == 1.0
    assert comparison["passed"] is True
    assert retrieval["summary"]["recall_at_k"] == {"1": 1.0, "3": 1.0, "5": 1.0}
    assert 'tests="3" failures="0"' in junit
