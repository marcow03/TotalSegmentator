import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree

from totalsegmentator.aorta_report.centerline import (
    get_distance,
    get_len_path_mm,
    get_normal_for_cl_point,
)
from totalsegmentator.aorta_report.geometry import (
    draw_plane,
    find_center,
    get_plane_diameters,
    get_region_diameters,
    get_region_diameters_pd,
)
from totalsegmentator.aorta_report.geometry_report import (
    get_2d_plane,
    keep_closest_blob,
    process_plane,
)
from totalsegmentator.aorta_report.utils import crop_to_bbox, get_bbox_around_center
from totalsegmentator.iliac_report.constants import (
    EDGE_EXCLUSION_MM,
    GAP_EXCLUSION_MM,
    NARROWING_RATIO,
    PROFILE_CROP_MM,
    SECTIONS,
)

LANDMARK_METRICS = (
    "area",
    "diameter",
    "diameter_pd",
    "start",
    "end",
    "start_pd",
    "end_pd",
)
SECTION_METRICS = (
    "length",
    "volume",
    "max_diameter",
    "min_diameter",
    "tortuosity_index",
    "excluded_regions",
)


def plane_intersection(mask, point, normal):
    """Cross-section of `mask` with the plane through `point`.

    Unlike aorta_report's get_plane_aorta_intersection the intersection is not
    dilated, which would add ~2 voxels to every diameter (+10-15% for iliacs).
    `draw_plane` is two voxels thick, so the cross-section has no holes.
    """
    intersection = (draw_plane(mask.shape, normal, point) > 0) & (mask > 0)
    return keep_closest_blob(intersection, point).astype(np.uint8)


def create_landmark_planes(landmarks, sides, aorta, spacing, logger):
    """Create measurement planes and return the largest preliminary diameter."""
    for number, landmark in landmarks.items():
        if landmark["empty"]:
            landmark["diameter_tmp"] = 0
            continue
        centerline = sides[landmark["cl_side"]]["centerline"]
        point = centerline[landmark["cl_idx"]].point
        normal = get_normal_for_cl_point(centerline, landmark["cl_idx"])
        mask = aorta if landmark["side"] is None else sides[landmark["side"]]["iliac"]
        plane = plane_intersection(mask, point, normal)
        landmark["cl_point"], landmark["roi"] = point, plane
        if not plane.any():
            logger.info(f"WARNING: for landmark {number} no vessel plane was found!")
            landmark["empty"] = True
            landmark["diameter_tmp"] = 0
            continue
        region = next(get_region_diameters(plane), None)
        if region is None:
            logger.info(f"WARNING: for landmark {number} no diameter was found!")
            landmark["diameter_tmp"] = 0
        else:
            _, start, end = region
            landmark["diameter_tmp"] = get_distance(start, end, spacing)
    return max(landmark["diameter_tmp"] for landmark in landmarks.values())


def measure_landmarks(landmarks, ct_img, affine, max_diameter, tmp_dir, logger, debug):
    for number, landmark in landmarks.items():
        logger.info(f"  Processing landmark {number}")
        if landmark["empty"]:
            landmark.update(dict.fromkeys(LANDMARK_METRICS))
            continue

        plane_img = nib.Nifti1Image(landmark["roi"], affine)
        plane_2d, ct_2d, _, _, inverse_affine = get_2d_plane(
            plane_img,
            ct_img,
            None,
            None,
            order=3,
            dtype=np.int32,
            addon=[max_diameter * 1.4] * 3,
        )
        # The plane is perpendicular to z after rotation. Project the thin slab
        # along z: a single slice misses parts of it if there is residual tilt.
        plane_volume = plane_2d.get_fdata() > 0.5
        center = find_center(plane_volume.astype(np.uint8), True)
        z_index = plane_volume.shape[2] // 2 if center is None else center[2]
        plane_data = binary_fill_holes(plane_volume.any(axis=2)).astype(np.uint8)
        plane_2d = nib.Nifti1Image(
            np.repeat(plane_data[:, :, None], plane_volume.shape[2], axis=2),
            plane_2d.affine,
            plane_2d.header,
        )
        plane_path = tmp_dir / f"plane_{number}.png"
        process_plane(ct_2d, plane_2d, None, plane_path, smooth=20)
        landmark["plane_img"] = str(plane_path)

        slice_spacing = np.asarray(plane_2d.header.get_zooms())[:2]
        landmark["area"] = (plane_data.sum() * np.prod(slice_spacing) / 100).round(1)

        diameter, perpendicular, endpoints, endpoints_mm = get_region_diameters_pd(
            plane_data, slice_spacing, inverse_affine, z=z_index
        )
        landmark["diameter"] = (diameter / 10).round(1)
        landmark["diameter_pd"] = (perpendicular / 10).round(1)
        (
            landmark["start"],
            landmark["end"],
            landmark["start_pd"],
            landmark["end_pd"],
        ) = endpoints_mm if endpoints_mm is not None else [None] * 4
        (
            landmark["start_voxel"],
            landmark["end_voxel"],
            landmark["start_pd_voxel"],
            landmark["end_pd_voxel"],
        ) = endpoints if endpoints is not None else [None] * 4

        if debug:
            nib.save(
                nib.Nifti1Image(landmark["roi"].astype(np.uint8), affine),
                tmp_dir / f"iliac_plane_lm{number}.nii.gz",
            )
    return landmarks


def diameter_profile(mask, centerline, spacing, crop_size_mm):
    """Major and perpendicular diameter (mm) at every centerline point.

    Returns a dict with the centerline `points` (voxel), their `distances` from
    the last centerline point (mm) and the `major` and `minor` diameters (mm).
    Points without a cross-section are NaN.
    """
    points = np.asarray([vertex.point for vertex in centerline], dtype=float).reshape(-1, 3)
    major = np.full(len(points), np.nan)
    minor = np.full(len(points), np.nan)
    if not len(points):
        return {"points": points, "distances": np.empty(0), "major": major, "minor": minor}
    radius = np.maximum(1, (np.asarray([crop_size_mm] * 3) / 2 / np.asarray(spacing)).astype(int))
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0) * np.asarray(spacing), axis=1)))
    )
    for idx, point in enumerate(points):
        bbox = get_bbox_around_center(point, radius, mask)
        cropped_point = point - np.asarray([axis[0] for axis in bbox])
        intersection = plane_intersection(
            crop_to_bbox(mask, bbox),
            cropped_point,
            get_normal_for_cl_point(centerline, idx),
        )
        if np.count_nonzero(intersection) < 5:
            continue
        diameter, perpendicular, axis, _ = get_plane_diameters(intersection, spacing)
        if axis is not None:
            major[idx], minor[idx] = diameter, perpendicular
    return {
        "points": points,
        "distances": cumulative[-1] - cumulative,
        "major": major,
        "minor": minor,
    }


def exclude_edges(profile, distance_mm):
    """Set diameters to NaN at both ends of the profile.

    At the bifurcation the plane cuts the junction of aorta and iliac artery,
    at the distal end the mask ends; neither gives a valid cross-section.
    """
    distances = profile["distances"]
    edge = (distances < distance_mm) | (distances > distances.max(initial=0) - distance_mm)
    profile["major"][edge] = np.nan
    profile["minor"][edge] = np.nan


def _exclude(profile, exclude, reason):
    """Set diameters to NaN where `exclude` is set and describe the excluded regions.

    Returns a list of {"reason", "start_mm", "end_mm"}, positions measured from the
    aortic bifurcation along the centerline.
    """
    profile["major"][exclude] = np.nan
    profile["minor"][exclude] = np.nan
    regions = []
    labels, count = ndimage.label(exclude)
    for label_idx in range(1, count + 1):
        distances = profile["distances"][labels == label_idx]
        regions.append(
            {
                "reason": reason,
                "start_mm": round(float(distances.min())),
                "end_mm": round(float(distances.max())),
            }
        )
    return regions


def exclude_near_gaps(profile, gap_points, spacing, distance_mm):
    """Exclude profile positions close to bridged segmentation gaps.

    The mask usually tapers off towards a gap, which would otherwise show up as
    an artificial narrowing.
    """
    if not len(gap_points) or not len(profile["points"]):
        return []
    spacing = np.asarray(spacing, dtype=float)
    distances, _ = cKDTree(np.asarray(gap_points) * spacing).query(profile["points"] * spacing)
    return _exclude(profile, distances < distance_mm, "segmentation gap")


def exclude_focal_narrowings(profile, ratio, distance_mm):
    """Exclude focal drops of the short-axis diameter below `ratio` x median.

    The iliac segmentation often narrows at the common/external iliac transition
    although the lumen is patent. Such narrowings are excluded, but reported, as
    they can also be a real stenosis.
    """
    minor = profile["minor"]
    valid = np.isfinite(minor) & (minor > 0)
    if not valid.any():
        return []
    narrow = valid & (minor < ratio * np.median(minor[valid]))
    if not narrow.any():
        return []
    narrow_distances = profile["distances"][narrow]
    near = np.abs(profile["distances"][:, None] - narrow_distances[None, :]).min(axis=1) < distance_mm
    return _exclude(profile, near, "focal narrowing (segmentation artifact or stenosis)")


def compute_profiles(sides, spacing, logger):
    """Diameter profiles of every available side along its iliac centerline.

    Every profile gets a list of `excluded` regions that were not measured.
    """
    profiles = {}
    for side, side_data in sides.items():
        if side_data is None:
            continue
        profile = diameter_profile(side_data["iliac"], side_data["resampled"], spacing, PROFILE_CROP_MM)
        exclude_edges(profile, EDGE_EXCLUSION_MM)
        profile["excluded"] = exclude_near_gaps(
            profile, side_data["gap_points"], spacing, GAP_EXCLUSION_MM
        ) + exclude_focal_narrowings(profile, NARROWING_RATIO, GAP_EXCLUSION_MM)
        for region in profile["excluded"]:
            logger.info(
                f"WARNING: {side} iliac artery {region['start_mm']}-{region['end_mm']}mm from the "
                f"bifurcation is not measured: {region['reason']}."
            )
        profiles[side] = profile
    return profiles


def tortuosity_index(centerline, spacing):
    """Centerline length divided by the straight distance between its ends."""
    if len(centerline) < 2:
        return None
    chord = get_distance(centerline[0].point, centerline[-1].point, spacing)
    if chord == 0:
        return None
    return round(float(get_len_path_mm(centerline, spacing) / chord), 2)


def measure_sections(sides, profiles, spacing, logger):
    """Per side: section from the aortic bifurcation to the distal iliac end.

    Maximum and minimum diameter are short-axis (perpendicular) diameters of
    the profile, which excludes the section ends and segmentation gaps.
    """
    section_stats = {}
    for name, side in SECTIONS.items():
        logger.info(f"  Processing section: {name}...")
        side_data = sides.get(side)
        if side_data is None:
            logger.info(f"WARNING: {name} is empty!")
            section_stats[name] = dict.fromkeys(SECTION_METRICS)
            continue
        section = side_data["centerline"][: side_data["bif_idx"] + 1]
        profile = profiles[side]
        minor = profile["minor"]
        valid = minor[np.isfinite(minor) & (minor > 0)]
        if not valid.size:
            logger.info(f"WARNING: no diameters found for {name}!")
        section_stats[name] = {
            "length": round(float(get_len_path_mm(section, spacing)) / 10, 1),
            "volume": round(float(side_data["iliac"].sum() * np.prod(spacing)) / 1000, 1),
            "max_diameter": round(float(valid.max()) / 10, 1) if valid.size else None,
            "min_diameter": round(float(valid.min()) / 10, 1) if valid.size else None,
            "tortuosity_index": tortuosity_index(section, spacing),
            "excluded_regions": profile["excluded"],
        }
    return section_stats
