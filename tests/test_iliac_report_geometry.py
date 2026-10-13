import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

from totalsegmentator.aorta_report.centerline import Vertex, get_len_path_mm
from totalsegmentator.iliac_report.centerline import (
    bridge_gaps,
    build_side_centerline,
    cut_aorta_above_iliacs,
)
from totalsegmentator.iliac_report.landmarks import place_landmarks
from totalsegmentator.iliac_report.measurements import (
    compute_profiles,
    create_landmark_planes,
    measure_landmarks,
    measure_sections,
    tortuosity_index,
)

SPACING = (0.8, 0.8, 0.8)
SHAPE = (120, 60, 200)
BIFURCATION = np.array([60.0, 30.0, 120.0])
DISTAL_ENDS = {"right": np.array([25.0, 30.0, 0.0]), "left": np.array([95.0, 30.0, 0.0])}
BULGE_FRACTION = 0.5  # bulge on the right iliac, halfway down


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def _segment_distance_mm(start, end):
    """Distance (mm) of every voxel to the segment start-end, plus the segment parameter."""
    grid = np.stack(np.indices(SHAPE), axis=-1).astype(float) * SPACING
    start, end = start * SPACING, end * SPACING
    direction = end - start
    t = np.clip(((grid - start) @ direction) / (direction @ direction), 0, 1)
    closest = start + t[..., None] * direction
    return np.linalg.norm(grid - closest, axis=-1), t


def _phantom(bulge=True, neck=False, gap=False):
    """Inverted Y: aorta (r=8mm) splitting into two iliac arteries (r=5mm).

    The right iliac artery has a local bulge (r=7mm) halfway down.
    """
    distance, _ = _segment_distance_mm(BIFURCATION, np.array([60.0, 30.0, SHAPE[2] - 1.0]))
    aorta = distance <= 8
    aorta[:, :, : int(BIFURCATION[2])] = False
    iliacs = {}
    for side, end in DISTAL_ENDS.items():
        distance, t = _segment_distance_mm(BIFURCATION, end)
        radius = np.full(SHAPE, 5.0)
        if bulge and side == "right":
            radius = 5.0 + 2.0 * np.exp(-(((t - BULGE_FRACTION) / 0.06) ** 2))
        if neck and side == "right":
            radius = 5.0 - 3.5 * np.exp(-(((t - BULGE_FRACTION) / 0.03) ** 2))
        iliac = (distance <= radius) & ~aorta
        if gap and side == "left":
            iliac &= np.abs(t - BULGE_FRACTION) > 0.03
        iliac[:, :, int(BIFURCATION[2]) :] = False
        iliacs[side] = iliac.astype(np.uint8)
    return aorta.astype(np.uint8), iliacs


@pytest.fixture(scope="module")
def phantom_report(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("iliac_geometry")
    aorta, iliacs = _phantom()
    logger = _Logger()
    affine = np.diag([*SPACING, 1.0])
    sides = {
        side: build_side_centerline(aorta, iliac, affine, SPACING, logger, side)
        for side, iliac in iliacs.items()
    }
    profiles = compute_profiles(sides, SPACING, logger)
    landmarks = place_landmarks(sides, profiles, SPACING, logger)
    max_diameter = create_landmark_planes(landmarks, sides, aorta, SPACING, logger)
    vessels = aorta | iliacs["right"] | iliacs["left"]
    ct_img = nib.Nifti1Image(vessels.astype(np.int16) * 300, affine)
    measure_landmarks(landmarks, ct_img, affine, max_diameter, tmp_dir, logger, False)
    section_stats = measure_sections(sides, profiles, SPACING, logger)
    assert not any("Bridging" in message for message in logger.messages)
    return {
        "sides": sides,
        "landmarks": landmarks,
        "section_stats": section_stats,
        "logger": logger,
    }


def _distance_mm(a, b):
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) * SPACING))


def test_bifurcation_is_found_at_the_split(phantom_report):
    for side_data in phantom_report["sides"].values():
        point = side_data["centerline"][side_data["bif_idx"]].point
        assert abs(point[2] - BIFURCATION[2]) * SPACING[2] < 5


def test_proximal_and_distal_landmarks_are_at_their_offsets(phantom_report):
    landmarks, sides = phantom_report["landmarks"], phantom_report["sides"]
    for side, (proximal, distal) in {"right": (2, 4), "left": (5, 7)}.items():
        centerline, bif_idx = sides[side]["centerline"], sides[side]["bif_idx"]
        proximal_idx = landmarks[proximal]["cl_idx"]
        distal_idx = landmarks[distal]["cl_idx"]
        assert get_len_path_mm(centerline[proximal_idx : bif_idx + 1], SPACING) == pytest.approx(10, abs=2)
        assert get_len_path_mm(centerline[: distal_idx + 1], SPACING) == pytest.approx(10, abs=2)


def test_max_landmark_is_on_the_bulge(phantom_report):
    landmark = phantom_report["landmarks"][3]
    bulge = BIFURCATION + BULGE_FRACTION * (DISTAL_ENDS["right"] - BIFURCATION)
    assert _distance_mm(landmark["cl_point"], bulge) < 8


def test_landmark_diameters_and_areas(phantom_report):
    """Plane measurements must not be inflated by dilating the cross-section."""
    landmarks = phantom_report["landmarks"]
    assert all(not landmark["empty"] for landmark in landmarks.values())
    expected_radius_mm = {1: 8, 2: 5, 3: 7, 4: 5, 5: 5, 6: 5, 7: 5}
    for number, radius in expected_radius_mm.items():
        for key in ("diameter", "diameter_pd"):
            assert landmarks[number][key] == pytest.approx(2 * radius / 10, abs=0.1)
        assert landmarks[number]["area"] == pytest.approx(np.pi * radius**2 / 100, rel=0.1)


def test_section_stats(phantom_report):
    right = phantom_report["section_stats"]["iliac_right"]
    left = phantom_report["section_stats"]["iliac_left"]
    expected_length = _distance_mm(BIFURCATION, DISTAL_ENDS["right"]) / 10
    for stats in (right, left):
        assert stats["length"] == pytest.approx(expected_length, abs=1.0)
        assert stats["min_diameter"] == pytest.approx(1.0, abs=0.1)
        assert stats["tortuosity_index"] == pytest.approx(1.0, abs=0.03)
        assert stats["volume"] > 0
    assert right["max_diameter"] == pytest.approx(1.4, abs=0.1)
    assert left["max_diameter"] == pytest.approx(1.0, abs=0.1)


def test_tortuosity_of_a_curved_centerline():
    straight = [Vertex((0, 0, z)) for z in range(100)]
    curved = [Vertex((10 * np.sin(z / 8), 0, z)) for z in range(100)]
    assert tortuosity_index(straight, (1, 1, 1)) == pytest.approx(1.0)
    assert tortuosity_index(curved, (1, 1, 1)) > 1.1
    assert tortuosity_index(straight[:1], (1, 1, 1)) is None


def test_missing_side_gives_empty_landmarks_and_sections():
    aorta, iliacs = _phantom(bulge=False)
    logger = _Logger()
    affine = np.diag([*SPACING, 1.0])
    sides = {
        "right": build_side_centerline(aorta, iliacs["right"], affine, SPACING, logger, "right"),
        "left": None,
    }
    profiles = compute_profiles(sides, SPACING, logger)
    landmarks = place_landmarks(sides, profiles, SPACING, logger)
    create_landmark_planes(landmarks, sides, aorta, SPACING, logger)
    section_stats = measure_sections(sides, profiles, SPACING, logger)
    assert not any(landmarks[number]["empty"] for number in (1, 2, 3, 4))
    assert all(landmarks[number]["empty"] for number in (5, 6, 7))
    assert all(value is None for value in section_stats["iliac_left"].values())
    assert section_stats["iliac_right"]["length"] > 0
    assert "WARNING: landmark 5 is empty because the left iliac artery is empty!" in logger.messages


def test_bridge_gaps_connects_close_fragments_and_drops_far_ones():
    mask = np.zeros((20, 20, 120), dtype=np.uint8)
    mask[8:12, 8:12, 0:40] = 1
    mask[8:12, 8:12, 46:80] = 1  # 6 voxel gap -> bridged
    mask[8:12, 8:12, 110:120] = 1  # 30 voxel gap -> dropped
    kept, bridged = bridge_gaps(mask, (1, 1, 1), max_gap_mm=20, min_size_mm3=10)
    assert kept[:, :, :80].sum() == mask[:, :, :80].sum()
    assert not kept[:, :, 110:].any()
    assert not kept[:, :, 41:45].any()
    assert bridged[:, :, 41:45].any()
    assert ndimage.label(bridged, structure=np.ones((3, 3, 3)))[1] == 1


def test_cut_aorta_above_iliacs():
    aorta = np.zeros((4, 4, 100), dtype=np.uint8)
    aorta[1:3, 1:3, 40:] = 1
    iliacs = np.zeros_like(aorta)
    iliacs[1:3, 1:3, :50] = 1
    cut = cut_aorta_above_iliacs(aorta, iliacs, 1.0, 20)
    assert cut[:, :, 69].any()
    assert not cut[:, :, 70:].any()
    assert not cut_aorta_above_iliacs(aorta, np.zeros_like(iliacs), 1.0, 20).any()


def test_focal_narrowings_and_gaps_are_excluded_and_reported():
    aorta, iliacs = _phantom(bulge=False, neck=True, gap=True)
    logger = _Logger()
    affine = np.diag([*SPACING, 1.0])
    sides = {
        side: build_side_centerline(aorta, iliac, affine, SPACING, logger, side)
        for side, iliac in iliacs.items()
    }
    profiles = compute_profiles(sides, SPACING, logger)
    section_stats = measure_sections(sides, profiles, SPACING, logger)
    half_length_mm = _distance_mm(BIFURCATION, DISTAL_ENDS["right"]) * BULGE_FRACTION
    for name, reason in (("iliac_right", "focal narrowing"), ("iliac_left", "segmentation gap")):
        stats = section_stats[name]
        assert stats["min_diameter"] == pytest.approx(1.0, abs=0.1)
        assert len(stats["excluded_regions"]) == 1
        region = stats["excluded_regions"][0]
        assert region["reason"].startswith(reason)
        assert region["start_mm"] < half_length_mm < region["end_mm"]
    assert any("is not measured: focal narrowing" in message for message in logger.messages)
