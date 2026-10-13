import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.aorta_report.utils import crop_to_masks
from totalsegmentator.iliac_report.centerline import cut_aorta_above_iliacs
from totalsegmentator.iliac_report.constants import AORTA_CONTEXT_MM, ILIAC_MASKS


ROIS = ("aorta", *ILIAC_MASKS.values())
ROI_FILES = tuple(f"{roi}.nii.gz" for roi in ROIS)

MODEL_RUNS = (
    (
        "Running TotalSegmentator - iliac arteries",
        "  Skipping TotalSeg (already exists)",
        ROI_FILES,
        ("-ta", "total", "-rs", *ROIS),
    ),
)


def _command(ct_path, output_dir, task_args):
    return [
        "TotalSegmentator",
        "-i",
        str(ct_path),
        "-o",
        str(output_dir),
        "-ns",
        "1",
        *task_args,
    ]


def _outputs_exist(output_dir, names):
    return all((output_dir / name).exists() for name in names)


def _check_outputs(output_dir, names, description):
    missing = [str(output_dir / name) for name in names if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{description} did not create the expected output files:\n  - "
            + "\n  - ".join(missing)
        )


def _run(command, description):
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")


def get_iliac_fast(ct_path, output_dir, logger):
    fast_dir = Path(output_dir) / "fast"
    if _outputs_exist(fast_dir, ROI_FILES):
        logger.info("  Skipping TotalSeg iliac fast (already exists)")
        return fast_dir
    task_args = ("-ta", "total", "-rs", *ROIS, "--fast")
    _run(_command(Path(ct_path), fast_dir, task_args), "TotalSegmentator iliac fast")
    _check_outputs(fast_dir, ROI_FILES, "TotalSegmentator iliac fast")
    return fast_dir


def crop_to_iliacs(ct_path, fast_dir, output_dir, logger):
    output_path = Path(output_dir) / "ct_cropped.nii.gz"
    if output_path.exists():
        logger.info("  already cropped (skipping)")
        return output_path
    fast_dir = Path(fast_dir)
    iliacs = [
        nib.as_closest_canonical(nib.load(fast_dir / f"{name}.nii.gz"))
        for name in ILIAC_MASKS.values()
    ]
    iliac_union = np.zeros(iliacs[0].shape, dtype=np.uint8)
    for iliac in iliacs:
        iliac_union[np.asanyarray(iliac.dataobj) > 0.5] = 1
    if not iliac_union.any():
        raise ValueError("No iliac arteries found in the field of view.")
    aorta_img = nib.as_closest_canonical(nib.load(fast_dir / "aorta.nii.gz"))
    aorta = cut_aorta_above_iliacs(
        np.asanyarray(aorta_img.dataobj) > 0.5,
        iliac_union,
        aorta_img.header.get_zooms()[2],
        AORTA_CONTEXT_MM + 20,
    )
    ct_img = nib.load(ct_path)
    print("Cropping...")
    print(f"  shape Before: {ct_img.shape}")
    ct_img = crop_to_masks(
        ct_img,
        [nib.Nifti1Image(iliac_union, iliacs[0].affine), nib.Nifti1Image(aorta, aorta_img.affine)],
        [20, 20, 20],
        dtype=np.int16,
    )
    print(f"  shape After: {ct_img.shape}")
    nib.save(ct_img, output_path)
    return output_path


def run_models_consecutive(ct_path, output_dir, logger):
    started = time.time()
    ct_path, output_dir = Path(ct_path), Path(output_dir)
    for status, skip_message, outputs, task_args in MODEL_RUNS:
        yield status
        if _outputs_exist(output_dir, outputs):
            logger.info(skip_message)
            continue
        _run(_command(ct_path, output_dir, task_args), status)
        _check_outputs(output_dir, outputs, status)
    print(f"Models done (took: {time.time() - started:.2f}s)")
    yield "Models done"
