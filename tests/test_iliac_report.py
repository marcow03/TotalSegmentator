import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent / "reference_files" / "iliac_report"
REFERENCE_JSON = REFERENCE_DIR / "iliac_report.json"
AORTA_REFERENCE_DIR = Path(__file__).parent / "reference_files" / "aorta_report"
EXAMPLE_NODEINFO = AORTA_REFERENCE_DIR / "example_nodeinfo.json"
EXAMPLE_CT = Path(
    os.environ.get(
        "ILIAC_REPORT_TEST_CT",
        Path(__file__).parents[2] / "data/totalsegmentator/Totalsegmentator_dataset_v300/s0733/ct.nii.gz",
    )
)

pytestmark = pytest.mark.skipif(not EXAMPLE_CT.exists(), reason=f"example CTA not found: {EXAMPLE_CT}")


def dicts_almost_equal(first, second):
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(dicts_almost_equal(first[key], second[key]) for key in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(dicts_almost_equal(left, right) for left, right in zip(first, second))
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return abs(first - second) <= 0.2
    return first == second


def remove_keys_with_prefix(value, prefixes):
    if isinstance(value, dict):
        return {key: remove_keys_with_prefix(item, prefixes) for key, item in value.items() if not any(key.startswith(prefix) for prefix in prefixes)}
    if isinstance(value, list):
        return [remove_keys_with_prefix(item, prefixes) for item in value]
    return value


def _run_report(tmp_path, extra_args):
    outputs = {
        "output_nifti": tmp_path / "iliac_report.nii.gz",
        "output_json": tmp_path / "iliac_report.json",
        "output_log": tmp_path / "iliac_report.log",
        "output_cpr": tmp_path / "iliac_cpr.png",
    }
    # model_runner calls the TotalSegmentator executable of this environment
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "totalsegmentator.bin.totalseg_iliac_report",
            "-i",
            str(EXAMPLE_CT),
            "-n",
            str(EXAMPLE_NODEINFO),
            "-o",
            str(outputs["output_nifti"]),
            "-j",
            str(outputs["output_json"]),
            "-l",
            str(outputs["output_log"]),
            "-c",
            str(outputs["output_cpr"]),
            "-tmp",
            str(tmp_path / "tmp"),
            "-t",
            "basic",
            "--debug",
            *extra_args,
        ],
        check=True,
        env=env,
    )
    return outputs


@pytest.fixture(scope="module")
def iliac_report_result(tmp_path_factory):
    return _run_report(tmp_path_factory.mktemp("iliac_report"), ["--run_models"])


@pytest.fixture(scope="module")
def iliac_report_precomputed_result(iliac_report_result, tmp_path_factory):
    models_dir = iliac_report_result["output_json"].parent / "tmp" / "models"
    return _run_report(tmp_path_factory.mktemp("iliac_report_rt"), ["-rt", str(models_dir)])


def test_iliac_report_files_exist(iliac_report_result):
    for path in iliac_report_result.values():
        assert path.is_file(), path


def test_iliac_report_nifti(iliac_report_result):
    image = nib.load(iliac_report_result["output_nifti"])
    data = np.asanyarray(image.dataobj)
    assert data.shape[2] == 24
    assert data.dtype.names == ("R", "G", "B")


def test_iliac_report_logs(iliac_report_result):
    logs = iliac_report_result["output_log"].read_text()
    for content in ("Processing landmark 1", "Processing landmark 7", "Processing section: iliac_left"):
        assert content in logs, f"required content not found in logs: {content}"


def test_iliac_report_json_is_plausible(iliac_report_result):
    with iliac_report_result["output_json"].open() as file:
        actual = json.load(file)
    assert set(actual) == {"landmarks", "section_stats", "metadata"}
    assert list(actual["landmarks"]) == [str(number) for number in range(1, 8)]
    for landmark in actual["landmarks"].values():
        assert 0.5 <= landmark["diameter"] <= 2.5
        assert 0.5 <= landmark["diameter_pd"] <= 2.5
    for section in actual["section_stats"].values():
        assert all(value is not None for value in section.values())
        assert 10 <= section["length"] <= 35
        assert 1.0 <= section["tortuosity_index"] <= 2.0


def test_iliac_report_json(iliac_report_result):
    with REFERENCE_JSON.open() as file:
        expected = json.load(file)
    with iliac_report_result["output_json"].open() as file:
        actual = json.load(file)
    expected = remove_keys_with_prefix(expected, ("start", "end"))
    actual = remove_keys_with_prefix(actual, ("start", "end"))
    expected["metadata"].pop("version", None)
    actual["metadata"].pop("version", None)
    assert dicts_almost_equal(expected, actual)


def test_iliac_report_with_precomputed_masks(iliac_report_result, iliac_report_precomputed_result):
    with iliac_report_result["output_json"].open() as file:
        expected = json.load(file)
    with iliac_report_precomputed_result["output_json"].open() as file:
        actual = json.load(file)
    assert dicts_almost_equal(
        remove_keys_with_prefix(expected, ("start", "end")),
        remove_keys_with_prefix(actual, ("start", "end")),
    )
