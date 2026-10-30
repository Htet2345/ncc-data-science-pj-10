import json
import subprocess
import sys

from mall_clustering.model import ROOT


def run(*args):
    return subprocess.run([sys.executable, "-m", "mall_clustering", *args], cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_fit_then_assign_in_separate_processes(tmp_path):
    result = run("train", "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rows"] == 200
    result = run("assign", "--income", "60", "--score", "50", "--model", str(tmp_path / "model.joblib"))
    assert result.returncode == 0, result.stderr
    assert isinstance(json.loads(result.stdout)["cluster_id"], int)


def test_missing_model_error(tmp_path):
    result = run("assign", "--income", "60", "--score", "50", "--model", str(tmp_path / "missing.joblib"))
    assert result.returncode == 2
    assert "training first" in result.stderr and "Traceback" not in result.stderr


def test_invalid_input_error():
    result = run("assign", "--income", "nan", "--score", "50")
    assert result.returncode == 2
    assert "finite" in result.stderr


def test_missing_data_error(tmp_path):
    result = run("train", "--data", str(tmp_path / "missing.csv"))
    assert result.returncode == 2
    assert "File not found" in result.stderr
