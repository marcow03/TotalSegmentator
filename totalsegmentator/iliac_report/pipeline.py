import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

from totalsegmentator.aorta_report.cpr import cpr
from totalsegmentator.aorta_report.utils import crop_to_masks, keep_largest_blob, lazy_load
from totalsegmentator.dicom_io import dcm_to_nifti
from totalsegmentator.iliac_report import VERSION
from totalsegmentator.iliac_report.centerline import build_side_centerline, cut_aorta_above_iliacs
from totalsegmentator.iliac_report.constants import (
    AORTA_CONTEXT_MM,
    ILIAC_MASKS,
    MIN_STRUCTURE_VOLUME_MM3,
    REPORT_FRAMES,
    REPORT_SMOOTHING,
)
from totalsegmentator.iliac_report.landmarks import place_landmarks
from totalsegmentator.iliac_report.measurements import (
    compute_profiles,
    create_landmark_planes,
    measure_landmarks,
    measure_sections,
)
from totalsegmentator.iliac_report.model_runner import (
    crop_to_iliacs,
    get_iliac_fast,
    run_models_consecutive,
)
from totalsegmentator.iliac_report.plotting import (
    plot_diameter_profiles,
    plot_iliac_cpr,
    plot_iliac_seg,
    vessel_image,
)
from totalsegmentator.iliac_report.rendering import (
    cleanup_workdir,
    prepare_results,
    render_report_image,
    serialize_outputs,
)
from totalsegmentator.rendering import fury_display_context
from totalsegmentator.reporting import setup_logger
from totalsegmentator.resampling import change_spacing
from totalsegmentator.serialization_utils import filestream_to_nifti


def _prepare_context(tmp_dir, logger, delete_tmp, delete_aux_files):
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
        delete_tmp = delete_aux_files = True
    tmp_dir = Path(tmp_dir)
    model_dir = tmp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if logger is None:
        logger = setup_logger(tmp_dir / "log.txt", name=f"totalseg_iliac_report.{tmp_dir}")
    return tmp_dir, model_dir, logger, delete_tmp, delete_aux_files


def _resolve_ct_path(ct_input, model_dir, tmp_dir, f_type, logger):
    ct_path = model_dir / "ct.nii.gz"
    if isinstance(ct_input, Path):
        return ct_input
    if isinstance(ct_input, nib.Nifti1Image):
        nib.save(ct_input, ct_path)
    elif f_type == "dicom":
        if ct_path.exists():
            logger.info("Found existing nifti files. Skipping conversion...")
        else:
            logger.info("Converting dicom to nifti...")
            dcm_to_nifti(ct_input, ct_path, tmp_dir=tmp_dir, verbose=True)
    else:
        nib.save(filestream_to_nifti(ct_input, gzipped=f_type == "niigz"), ct_path)
    return ct_path


def _run_segmentation_models(ct_path, model_dir, logger, test):
    fast_dir = get_iliac_fast(ct_path, model_dir, logger)
    ct_path = crop_to_iliacs(ct_path, fast_dir, model_dir, logger)
    new_spacing = 1.5 if test == "basic" else 0.8
    ct_img = change_spacing(nib.load(ct_path), new_spacing, order=3, dtype=np.int16)
    nib.save(ct_img, ct_path)
    print(f"Resampled CT to {new_spacing}mm: {ct_img.shape}")
    return ct_path, run_models_consecutive(ct_path, model_dir, logger)


def _validate_inputs(rois_totalseg_dir):
    if rois_totalseg_dir is None:
        raise FileNotFoundError("--rois_totalseg is required when --run_models is not used.")
    required = [rois_totalseg_dir / "aorta.nii.gz"]
    required.extend(rois_totalseg_dir / f"{name}.nii.gz" for name in ILIAC_MASKS.values())
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Iliac report requires precomputed aorta and iliac artery masks when "
            "--run_models is not used.\nMissing files:\n  - "
            + "\n  - ".join(str(path) for path in missing)
        )


def _load_report_data(ct_path, rois_totalseg_dir, tmp_dir, logger, debug):
    _validate_inputs(rois_totalseg_dir)
    iliac_imgs = {
        side: nib.as_closest_canonical(nib.load(rois_totalseg_dir / f"{name}.nii.gz"))
        for side, name in ILIAC_MASKS.items()
    }
    reference = next(iter(iliac_imgs.values()))
    iliac_union = np.zeros(reference.shape, dtype=np.uint8)
    for image in iliac_imgs.values():
        iliac_union[np.asanyarray(image.dataobj) > 0.5] = 1
    if not iliac_union.any():
        raise ValueError("No iliac arteries found in the field of view.")

    # Reference grid: iliac arteries + the lower aorta, 0.8mm isotropic.
    aorta_img = nib.as_closest_canonical(nib.load(rois_totalseg_dir / "aorta.nii.gz"))
    aorta_crop = cut_aorta_above_iliacs(
        np.asanyarray(aorta_img.dataobj) > 0.5,
        iliac_union,
        aorta_img.header.get_zooms()[2],
        AORTA_CONTEXT_MM,
    )
    reference_img = crop_to_masks(
        nib.Nifti1Image(iliac_union, reference.affine),
        [nib.Nifti1Image(iliac_union, reference.affine), nib.Nifti1Image(aorta_crop, aorta_img.affine)],
        [20, 20, 20],
        dtype=np.uint8,
    )
    reference_img = change_spacing(reference_img, 0.8, order=0)
    affine, spacing = reference_img.affine, reference_img.header.get_zooms()

    ct_img, _ = lazy_load(ct_path, reference_img, tmp_dir, order=3, is_mask=False)
    _, aorta = lazy_load(
        rois_totalseg_dir / "aorta.nii.gz", reference_img, tmp_dir, order=0, is_mask=True
    )
    iliacs = {}
    for side, name in ILIAC_MASKS.items():
        # Not loaded as mask: keep_largest_blob would drop iliac fragments
        # that are separated by segmentation gaps. Gaps are bridged later.
        _, data = lazy_load(
            rois_totalseg_dir / f"{name}.nii.gz",
            reference_img,
            tmp_dir,
            order=0,
            dtype=np.uint8,
            is_mask=False,
        )
        iliacs[side] = binary_fill_holes(data > 0).astype(np.uint8)
    aorta = cut_aorta_above_iliacs(
        aorta, iliacs["right"] | iliacs["left"], spacing[2], AORTA_CONTEXT_MM
    )
    aorta = binary_fill_holes(keep_largest_blob(aorta)).astype(np.uint8)
    if aorta.sum() * np.prod(spacing) < MIN_STRUCTURE_VOLUME_MM3:
        logger.info("WARNING: Aorta is (almost) empty! The bifurcation cannot be found.")
    for side, data in iliacs.items():
        if data.sum() * np.prod(spacing) < MIN_STRUCTURE_VOLUME_MM3:
            logger.info(f"WARNING: {side} iliac artery is (almost) empty!")
            iliacs[side] = np.zeros_like(data)
    if debug:
        nib.save(nib.Nifti1Image(aorta, affine), tmp_dir / "aorta_cropped.nii.gz")
    return {
        "ct_img": ct_img,
        "aorta": aorta,
        "iliacs": iliacs,
        "affine": affine,
        "spacing": spacing,
    }


def _build_centerlines(data, tmp_dir, logger, debug):
    sides = {}
    for side, iliac in data["iliacs"].items():
        if not iliac.any() or not data["aorta"].any():
            sides[side] = None
            continue
        sides[side] = build_side_centerline(
            data["aorta"], iliac, data["affine"], data["spacing"], logger, side, debug
        )
        if debug and sides[side] is not None:
            nib.save(
                nib.Nifti1Image(sides[side]["centerline_image"].astype(np.uint8), data["affine"]),
                tmp_dir / f"centerline_{side}.nii.gz",
            )
    return sides


def _create_cpr(data, sides, profiles, landmarks, max_diameter, cpr_path, metadata, tmp_dir, debug, logger):
    cprs = {}
    for side, side_data in sides.items():
        if side_data is None:
            logger.info(f"WARNING: Skipping {side} CPR because the centerline is empty.")
            continue
        started = time.time()
        ct_cpr, mask_cpr, cpr_info = cpr(
            data["ct_img"],
            vessel_image(side_data, data["aorta"], data["affine"]),
            side_data["centerline"],
            max_diameter,
            fast=debug,
            debug=debug,
            return_info=True,
        )
        logger.info(f"  {side} CPR took {time.time() - started:.2f}s")
        cprs[side] = (ct_cpr, mask_cpr, cpr_info, side_data["centerline"])
    if not cprs:
        logger.info("WARNING: Skipping CPR because no centerline is available.")
        return
    plot_iliac_cpr(cprs, profiles, landmarks, data["affine"], cpr_path, metadata, tmp_dir)


def create_iliac_report(
    ct_bytes,
    rois_totalseg_dir,
    metadata,
    tmp_dir,
    logger,
    delete_tmp=True,
    delete_aux_files=True,
    cpr_path=None,
    test="None",
    debug=False,
    run_models=False,
    f_type="nii",
    version=VERSION,
):
    """Generate an iliac artery report and yield progress dictionaries."""
    yield {"id": 1, "progress": 2, "status": "Loading code"}
    tmp_dir, model_dir, logger, delete_tmp, delete_aux_files = _prepare_context(
        tmp_dir, logger, delete_tmp, delete_aux_files
    )
    completed = False
    try:
        ct_path = _resolve_ct_path(ct_bytes, model_dir, tmp_dir, f_type, logger)
        if run_models:
            yield {"id": 2, "progress": 5, "status": "Running models"}
            ct_path, statuses = _run_segmentation_models(ct_path, model_dir, logger, test)
            for index, status in enumerate(statuses):
                yield {"id": 3, "progress": 10 + index * 2, "status": status}
            rois_totalseg_dir = model_dir

        rois_totalseg_dir = Path(rois_totalseg_dir) if rois_totalseg_dir is not None else None
        yield {"id": 4, "progress": 20, "status": "Loading data"}
        data = _load_report_data(ct_path, rois_totalseg_dir, tmp_dir, logger, debug)

        yield {"id": 5, "progress": 25, "status": "Creating centerlines"}
        started = time.time()
        sides = _build_centerlines(data, tmp_dir, logger, debug)
        logger.info(f"  centerlines took {time.time() - started:.2f}s")
        profiles = compute_profiles(sides, data["spacing"], logger)
        landmarks = place_landmarks(sides, profiles, data["spacing"], logger)

        yield {"id": 6, "progress": 35, "status": "Getting max diameter"}
        max_diameter = create_landmark_planes(
            landmarks, sides, data["aorta"], data["spacing"], logger
        )
        measure_landmarks(
            landmarks, data["ct_img"], data["affine"], max_diameter, tmp_dir, logger, debug
        )
        section_stats = measure_sections(sides, profiles, data["spacing"], logger)

        if cpr_path is not None:
            logger.info("Calculating CPR...")
            _create_cpr(
                data, sides, profiles, landmarks, max(max_diameter, 20), cpr_path, metadata, tmp_dir, debug, logger
            )

        yield {"id": 7, "progress": 85, "status": "Generating report images"}
        profile_img = plot_diameter_profiles(
            sides, profiles, landmarks, tmp_dir / "diameter_profiles.png"
        )
        with fury_display_context():
            plot_iliac_seg(
                data["aorta"], sides, landmarks, tmp_dir, REPORT_SMOOTHING, REPORT_FRAMES
            )

        yield {"id": 8, "progress": 90, "status": "Generating report html"}
        report_image = render_report_image(landmarks, section_stats, profile_img, metadata, tmp_dir)
        results = prepare_results(landmarks, section_stats, metadata)
        yield {"id": 9, "progress": 95, "status": "Combine masks for output"}
        serialized_report, serialized_masks = serialize_outputs(
            report_image, results, tmp_dir, logger
        )
        cleanup_workdir(tmp_dir, delete_tmp, delete_aux_files)
        completed = True
        yield {"id": 10, "progress": 96, "status": "Returning data"}
        yield {
            "id": 11,
            "progress": 100,
            "status": "Done",
            "report_img": serialized_report,
            "report_json": results,
            "masks": serialized_masks,
        }
    finally:
        if not completed and delete_tmp and tmp_dir.exists():
            cleanup_workdir(tmp_dir, delete_tmp=True, delete_aux_files=False)


__all__ = ["create_iliac_report", "VERSION"]
