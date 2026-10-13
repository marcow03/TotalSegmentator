import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.spatial import cKDTree

from totalsegmentator.aorta_report.centerline import (
    Vertex,
    get_centerline,
    reorder_centerline,
    resample_cl_to_same_distance,
)
from totalsegmentator.aorta_report.geometry import smooth_mask
from totalsegmentator.iliac_report.constants import (
    CENTERLINE_SMOOTHING,
    MAX_GAP_MM,
    MIN_FRAGMENT_MM3,
)


def cut_aorta_above_iliacs(aorta, iliacs, spacing_z, context_mm):
    """Keep only the aorta up to `context_mm` above the highest iliac voxel.

    Arrays must be in canonical (RAS) orientation so that the last axis points
    cranially.
    """
    aorta = np.asarray(aorta) > 0
    iliac_z = np.where(np.asarray(iliacs) > 0)[2]
    if not iliac_z.size:
        return np.zeros(aorta.shape, dtype=np.uint8)
    top = int(iliac_z.max() + np.ceil(context_mm / spacing_z))
    result = np.zeros(aorta.shape, dtype=np.uint8)
    result[:, :, : top + 1] = aorta[:, :, : top + 1]
    return result


def _draw_tube(mask, start, end, radius):
    length = int(np.ceil(np.linalg.norm(end - start))) + 1
    limit = np.asarray(mask.shape) - 1
    for point in np.linspace(start, end, length):
        low = np.clip(np.floor(point - radius).astype(int), 0, limit)
        high = np.clip(np.ceil(point + radius).astype(int), 0, limit)
        mask[low[0] : high[0] + 1, low[1] : high[1] + 1, low[2] : high[2] + 1] = 1


def bridge_gaps(mask, spacing, max_gap_mm, min_size_mm3, logger=None, name="mask"):
    """Connect fragments of a vessel mask that are separated by small gaps.

    Segmentations sometimes break a vessel into several pieces. Components
    smaller than `min_size_mm3` or further than `max_gap_mm` from the main
    vessel are dropped. The others are attached by a thin tube.

    Returns (kept, bridged): `kept` holds the kept components without any
    added voxels (for measurements), and `bridged` is the connected mask
    including the tubes (for the centerline).
    """
    labels, count = ndimage.label(np.asarray(mask) > 0, structure=np.ones((3, 3, 3)))
    if count == 0:
        empty = np.zeros(labels.shape, dtype=np.uint8)
        return empty, empty.copy()
    sizes = np.bincount(labels.ravel())[1:] * np.prod(spacing)
    spacing = np.asarray(spacing, dtype=float)
    main = int(np.argmax(sizes)) + 1
    kept = labels == main
    bridged = kept.copy()
    candidates = [idx + 1 for idx in np.argsort(sizes)[::-1] if idx + 1 != main]
    while candidates:
        main_points = np.argwhere(kept)
        tree = cKDTree(main_points * spacing)
        best = None
        for component in candidates:
            if sizes[component - 1] < min_size_mm3:
                continue
            points = np.argwhere(labels == component)
            distances, indexes = tree.query(points * spacing)
            closest = int(np.argmin(distances))
            if best is None or distances[closest] < best[0]:
                best = (distances[closest], component, points[closest], main_points[indexes[closest]])
        if best is None or best[0] > max_gap_mm:
            break
        distance, component, start, end = best
        if logger is not None:
            logger.info(f"  Bridging a {distance:.1f}mm gap in {name}")
        kept |= labels == component
        bridged |= labels == component
        _draw_tube(bridged, start.astype(float), end.astype(float), radius=2)
        candidates.remove(component)
    dropped = sum(sizes[component - 1] for component in candidates)
    if dropped and logger is not None:
        logger.info(f"  Removed {dropped:.0f}mm3 of disconnected fragments from {name}")
    return kept.astype(np.uint8), bridged.astype(np.uint8)


def smooth_centerline(centerline, sigma):
    """Remove the voxel staircase of a skeleton path.

    Arc lengths of an unsmoothed skeleton overestimate the length of oblique
    vessels by up to ~8%.
    """
    if len(centerline) < 3:
        return centerline
    points = np.asarray([vertex.point for vertex in centerline], dtype=float)
    smoothed = gaussian_filter1d(points, sigma=sigma, axis=0, mode="nearest")
    smoothed[0], smoothed[-1] = points[0], points[-1]
    return [Vertex(point) for point in smoothed]


def _point_in_mask(mask, point):
    index = np.clip(np.rint(point).astype(int), 0, np.asarray(mask.shape) - 1)
    return bool(mask[tuple(index)])


def find_bifurcation_index(centerline, aorta, tolerance=2):
    """Return the first centerline index below the aorta.

    The centerline runs from the distal iliac end (index 0) to the aortic end
    (last index). Walking from the aortic end, the bifurcation is the first
    point after which `tolerance` consecutive points lie outside the aorta.
    """
    if not centerline or not _point_in_mask(aorta, centerline[-1].point):
        return None
    outside = 0
    for idx in range(len(centerline) - 1, -1, -1):
        outside = 0 if _point_in_mask(aorta, centerline[idx].point) else outside + 1
        if outside == tolerance:
            return idx + tolerance - 1
    return None


def build_side_centerline(aorta, iliac, affine, spacing, logger, side, debug=False):
    """Centerline through aorta + one iliac artery, from distal iliac end to aorta.

    Gaps in the segmentation are bridged for the centerline only. The returned
    `iliac` mask contains the iliac voxels connected to the vessel tree, without
    any bridging voxels. Returns None if no usable centerline was found.
    """
    kept, vessel = bridge_gaps(
        (aorta > 0) | (iliac > 0), spacing, MAX_GAP_MM, MIN_FRAGMENT_MM3, logger, f"{side} iliac artery"
    )
    dilated = binary_dilation(vessel, iterations=2)
    vessel_smooth = smooth_mask(dilated, sigma=3)
    vessel_for_centerline = vessel_smooth if vessel_smooth.any() else dilated.astype(np.uint8)
    try:
        centerline_image, centerline = get_centerline(vessel_for_centerline, debug=debug)
    except ValueError:
        logger.info(f"WARNING: Unable to create a centerline for the {side} iliac artery.")
        return None
    centerline = smooth_centerline(reorder_centerline(centerline), CENTERLINE_SMOOTHING)
    if len(centerline) < 2:
        logger.info(f"WARNING: Centerline of the {side} iliac artery is too short.")
        return None

    dilated_iliac = binary_dilation(iliac > 0, iterations=3)
    if not _point_in_mask(dilated_iliac, centerline[0].point):
        logger.info(
            f"WARNING: Distal end of the {side} centerline is not in the iliac artery."
        )
    bif_idx = find_bifurcation_index(centerline, aorta > 0)
    if bif_idx is None:
        logger.info(f"WARNING: Could not find the aortic bifurcation on the {side} centerline.")
        return None
    return {
        "iliac": ((iliac > 0) & (kept > 0)).astype(np.uint8),
        "gap_points": np.argwhere((vessel > 0) & (kept == 0)),
        "centerline": centerline,
        "centerline_image": centerline_image,
        # iliac part only (distal end -> bifurcation), 1 mm steps
        "resampled": resample_cl_to_same_distance(centerline[: bif_idx + 1], affine, 1),
        "bif_idx": bif_idx,
    }
