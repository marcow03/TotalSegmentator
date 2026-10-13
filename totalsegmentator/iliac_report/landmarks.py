import numpy as np
from scipy.ndimage import median_filter

from totalsegmentator.aorta_report.centerline import (
    get_mid_of_points,
    get_next_point_by_distance,
)
from totalsegmentator.iliac_report.constants import (
    BIFURCATION_OFFSET_MM,
    DISTAL_OFFSET_MM,
    PROXIMAL_OFFSET_MM,
    SIDES,
    create_landmarks,
)


def _landmark_number(side, kind):
    offset = {"right": 2, "left": 5}[side]
    return offset + ("proximal", "max", "distal").index(kind)


def max_short_axis_index(centerline, profile, start_idx, end_idx):
    """Centerline index with the largest short-axis diameter in [start_idx, end_idx].

    The short-axis (perpendicular) diameter is used because branch origins and
    oblique planes inflate the major diameter. A 3-point median suppresses
    single-plane outliers.
    """
    if profile is None or not len(profile["points"]):
        return None
    cl_points = np.asarray([vertex.point for vertex in centerline], dtype=float)
    cl_indexes = np.asarray(
        [int(np.argmin(np.linalg.norm(cl_points - point, axis=1))) for point in profile["points"]]
    )
    minor = median_filter(np.nan_to_num(profile["minor"], nan=0.0), size=3, mode="nearest")
    inside = (cl_indexes >= start_idx) & (cl_indexes <= end_idx) & (minor > 0)
    if not inside.any():
        return None
    return int(cl_indexes[np.flatnonzero(inside)[np.argmax(minor[inside])]])


def place_landmarks(sides, profiles, spacing, logger):
    """Place landmarks on the per-side centerlines.

    sides: {side: centerline dict from build_side_centerline or None}
    profiles: {side: diameter profile from measurements.compute_profiles}

    Every landmark gets the centerline it belongs to (`cl_side`) and its index
    on that centerline (`cl_idx`). Landmarks that cannot be placed are empty.
    """
    landmarks = create_landmarks()

    def set_index(number, side, index, reason=None):
        landmark = landmarks[number]
        if index is None:
            landmark["empty"] = True
            logger.info(
                f"WARNING: landmark {number} is empty because {reason}!"
                if reason
                else f"WARNING: Could not place landmark {number} on the centerline!"
            )
            return
        landmark["cl_side"] = side
        landmark["cl_idx"] = int(index)

    # Aortic bifurcation: measured on the right centerline, left as fallback.
    anchor_side = next((side for side in SIDES if sides.get(side) is not None), None)
    if anchor_side is None:
        set_index(1, None, None, "no iliac centerline was found")
    else:
        anchor = sides[anchor_side]
        set_index(
            1,
            anchor_side,
            get_next_point_by_distance(
                anchor["centerline"], anchor["bif_idx"], BIFURCATION_OFFSET_MM, spacing
            ),
        )

    for side in SIDES:
        proximal_nr = _landmark_number(side, "proximal")
        max_nr = _landmark_number(side, "max")
        distal_nr = _landmark_number(side, "distal")
        side_data = sides.get(side)
        if side_data is None:
            for number in (proximal_nr, max_nr, distal_nr):
                set_index(number, side, None, f"the {side} iliac artery is empty")
            continue
        centerline, bif_idx = side_data["centerline"], side_data["bif_idx"]
        proximal_idx = get_next_point_by_distance(centerline, bif_idx, -PROXIMAL_OFFSET_MM, spacing)
        distal_idx = get_next_point_by_distance(centerline, 0, DISTAL_OFFSET_MM, spacing)
        if proximal_idx is not None and distal_idx is not None and distal_idx >= proximal_idx:
            logger.info(
                f"WARNING: {side} iliac artery is too short for proximal and distal landmarks!"
            )
            proximal_idx = distal_idx = None
        set_index(proximal_nr, side, proximal_idx)
        set_index(distal_nr, side, distal_idx)
        if proximal_idx is None or distal_idx is None:
            set_index(max_nr, side, None, f"the {side} proximal or distal landmark is empty")
            continue

        max_idx = max_short_axis_index(centerline, profiles.get(side), distal_idx, proximal_idx)
        if max_idx is None:
            logger.info(
                f"WARNING: {side} iliac section is too short; using its midpoint as max."
            )
            max_idx = get_mid_of_points(centerline, distal_idx, proximal_idx)
        set_index(max_nr, side, max_idx)

    for landmark in landmarks.values():
        if landmark["empty"]:
            landmark["cl_idx"] = None
    return landmarks
