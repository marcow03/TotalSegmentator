SIDES = ("right", "left")

ILIAC_MASKS = {
    "right": "iliac_artery_right",
    "left": "iliac_artery_left",
}

# index: (name, side, display_name)
LANDMARK_DEFINITIONS = {
    1: ("aortic_bifurcation", None, "aortic bifurcation"),
    2: ("right_proximal", "right", "R iliac prox."),
    3: ("right_max", "right", "R iliac max."),
    4: ("right_distal", "right", "R iliac distal"),
    5: ("left_proximal", "left", "L iliac prox."),
    6: ("left_max", "left", "L iliac max."),
    7: ("left_distal", "left", "L iliac distal"),
}

SECTIONS = {
    "iliac_right": "right",
    "iliac_left": "left",
}

AORTA_CONTEXT_MM = 40  # aorta kept above the top of the iliac arteries
BIFURCATION_OFFSET_MM = 10  # landmark 1: cranial of the bifurcation, on the aorta
PROXIMAL_OFFSET_MM = 10  # distal of the bifurcation
DISTAL_OFFSET_MM = 10  # proximal of the distal end of the iliac centerline
EDGE_EXCLUSION_MM = 5  # ignored at both section ends for min/max diameter
MIN_STRUCTURE_VOLUME_MM3 = 200
MAX_GAP_MM = 20  # larger gaps in the segmentation are not bridged
MIN_FRAGMENT_MM3 = 100  # smaller disconnected fragments are removed
GAP_EXCLUSION_MM = 10  # no profile measurements this close to a gap or focal narrowing
NARROWING_RATIO = 0.5  # short-axis diameter below ratio x median: focal narrowing
PROFILE_CROP_MM = 80
CENTERLINE_SMOOTHING = 3  # gaussian sigma in centerline points

REPORT_FRAMES = 24
REPORT_SMOOTHING = 100


def create_landmarks():
    return {
        index: {
            "name": name,
            "display_name": display_name,
            "side": side,
            "cl_idx": None,
            "empty": False,
        }
        for index, (name, side, display_name) in LANDMARK_DEFINITIONS.items()
    }
