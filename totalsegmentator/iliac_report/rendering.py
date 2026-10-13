import shutil
from importlib import resources

import nibabel as nib
import numpy as np

from totalsegmentator.aorta_report.nifti import combine_as_nifti
from totalsegmentator.iliac_report.constants import ILIAC_MASKS, REPORT_FRAMES
from totalsegmentator.reporting import generate_html
from totalsegmentator.serialization_utils import serialize_and_compress


def render_report_image(landmarks, section_stats, profile_img, metadata, tmp_dir):
    assets_dir = resources.files("totalsegmentator").joinpath("resources/iliac_report")
    slices = []
    for frame_index in range(REPORT_FRAMES):
        output_path = tmp_dir / f"iliac_report_frontpage_{frame_index:02d}.png"
        generate_html(
            str(assets_dir),
            "report_template_frontpage.html",
            {
                "landmarks": landmarks,
                "section_stats": section_stats,
                "profile_img": str(profile_img),
                "preview_3d": str(tmp_dir / f"preview_3d_rotating_{frame_index:06d}.png"),
                "metadata": metadata,
            },
            output_path,
            width=900,
            png_logo_hosts=("rigi", "rndapollolp01.uhbs.ch"),
        )
        slices.append(output_path)
    return combine_as_nifti(slices)


def prepare_results(landmarks, section_stats, metadata):
    landmarks = dict(sorted(landmarks.items()))
    for landmark in landmarks.values():
        for key in (
            "roi",
            "plane_img",
            "cl_idx",
            "cl_point",
            "cl_side",
            "diameter_tmp",
            "color",
            "txt_offset",
            "empty",
            "side",
        ):
            landmark.pop(key, None)
        landmark["name"] = landmark.pop("display_name")
    return {"landmarks": landmarks, "section_stats": section_stats, "metadata": metadata}


def serialize_outputs(report_image, results, tmp_dir, logger):
    masks = {}
    for name in ILIAC_MASKS.values():
        mask_path = tmp_dir / f"{name}.nii.gz"
        if mask_path.exists():
            image = nib.load(mask_path)
            masks[name] = nib.Nifti1Image(
                image.get_fdata(dtype=np.float32).astype(np.uint8),
                image.affine,
                image.header,
            )
        else:
            logger.info(f"WARNING: Mask {name} not found for final output. Skipping.")
    return serialize_and_compress(report_image), serialize_and_compress(masks)


def cleanup_workdir(tmp_dir, delete_tmp, delete_aux_files):
    if delete_aux_files and tmp_dir.exists():
        for path in tmp_dir.glob("*.png"):
            if not path.name.startswith("cpr"):
                path.unlink()
        for path in tmp_dir.glob("*.html"):
            path.unlink()
    if delete_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
