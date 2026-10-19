from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
from PIL import Image
from scipy.ndimage import binary_dilation

from totalsegmentator.aorta_report.plotting import (
    _add_surface,
    _get_cpr_longitudinal_views,
    _get_landmark_positions,
    _plot_mask_contours,
    _render_preview_scene,
    _json_values,
    _style_diameter_profile,
    colors,
    save_cpr_view,
    save_profile_json,
)
from totalsegmentator.aorta_report.report_layout import compose_report_image
from totalsegmentator.iliac_report.constants import SIDES

REPORT_TITLE = "Iliac Artery Report"
SIDE_COLORS = {"right": "#f7c66a", "left": "#6ab0f7"}
LANDMARK_COLORS = (
    "red_light",
    "green",
    "yellow",
    "blue_light",
    "orange",
    "pink",
    "purple",
)


def plot_iliac_seg(aorta, sides, landmarks, output_path, smoothing=20, nr_frames=12):
    """Render aorta, both iliac arteries, centerlines and landmark planes."""
    from fury import window

    from totalsegmentator.rendering import plot_mask, text

    for number, landmark in landmarks.items():
        landmark["color"] = colors[LANDMARK_COLORS[(number - 1) % len(LANDMARK_COLORS)]]
        landmark["txt_offset"] = np.array([-20, 0, 0])

    surface_kwargs = dict(affine=np.eye(4), x_current=0, y_current=0, orientation="sagittal")
    scene = window.Scene()
    try:
        has_actor = _add_surface(
            scene, plot_mask, aorta, smoothing=smoothing,
            color=colors["gray_light"], opacity=0.3, **surface_kwargs,
        )
        for side_data in sides.values():
            if side_data is None:
                continue
            has_actor = _add_surface(
                scene, plot_mask, side_data["iliac"], smoothing=smoothing,
                color=colors["gray_light"], opacity=0.3, **surface_kwargs,
            ) or has_actor
            has_actor = _add_surface(
                scene, plot_mask, side_data["centerline_image"], smoothing=0,
                color=colors["red"], opacity=1.0, **surface_kwargs,
            ) or has_actor
        for number, landmark in landmarks.items():
            if landmark["empty"]:
                continue
            # dilated for display only, otherwise the plane is hidden inside the vessel
            has_actor = _add_surface(
                scene, plot_mask, binary_dilation(landmark["roi"], iterations=2).astype(np.uint8),
                smoothing=smoothing,
                color=landmark["color"], opacity=1.0, **surface_kwargs,
            ) or has_actor
            size = aorta.shape
            x, y, z = landmark["cl_point"]
            x = size[0] - x
            x, y, z = y, z, x
            x = size[1] - x
            position = np.array([x, y, z]) + landmark["txt_offset"]
            scene.add(text(number, position, (8, 8, 8), colors["white"]))
        _render_preview_scene(
            scene, Path(output_path) / "preview_3d_rotating_", (700, 900), nr_frames, has_actor
        )
    finally:
        scene.clear()


def _profile_landmark_distances(landmarks, side, profile, centerline):
    """Distance from the aortic bifurcation (mm) of the landmarks of one side."""
    result = []
    if profile is None or not len(profile["points"]):
        return result
    for number, landmark in landmarks.items():
        if landmark["empty"] or landmark["side"] != side:
            continue
        point = centerline[landmark["cl_idx"]].point
        closest = int(np.argmin(np.linalg.norm(profile["points"] - point, axis=1)))
        result.append((number, profile["distances"][closest]))
    return result


def _save_diameter_profiles_json(sides, profiles, landmarks, output_path):
    data = {"landmarks": []}
    for side in SIDES:
        profile = profiles.get(side)
        if profile is None or not len(profile["distances"]):
            continue
        data[side] = {
            "distances": _json_values(profile["distances"]),
            "minor": _json_values(profile["minor"]),
            "major": _json_values(profile["major"]),
            "excluded": [
                {"start_mm": float(r["start_mm"]), "end_mm": float(r["end_mm"]), "reason": r["reason"]}
                for r in profile.get("excluded", [])
            ],
        }
        for number, distance in _profile_landmark_distances(landmarks, side, profile, sides[side]["centerline"]):
            data["landmarks"].append({
                "number": number,
                "name": landmarks[number].get("display_name", landmarks[number].get("name")),
                "side": side,
                "distance_mm": round(float(distance), 2),
            })
    save_profile_json(output_path, data)


def plot_diameter_profiles(sides, profiles, landmarks, output_path):
    """Short- and long-axis diameter along both iliac arteries."""
    _save_diameter_profiles_json(sides, profiles, landmarks, Path(output_path).with_suffix(".json"))
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 3.0), facecolor="black")
    max_diameter = 10.0
    for side in SIDES:
        profile = profiles.get(side)
        if profile is None or not len(profile["distances"]):
            continue
        color = SIDE_COLORS[side]
        for region in profile.get("excluded", []):
            ax.axvspan(region["start_mm"], region["end_mm"], color=color, alpha=0.12, linewidth=0)
        ax.plot(profile["distances"], profile["minor"], color=color, linewidth=1.8,
                label=f"{side} (short axis)")
        ax.plot(profile["distances"], profile["major"], color=color, linewidth=0.8,
                alpha=0.5, linestyle="--", label=f"{side} (long axis)")
        if np.isfinite(profile["minor"]).any():
            max_diameter = max(max_diameter, float(np.nanmax(profile["minor"])))
        for number, distance in _profile_landmark_distances(
            landmarks, side, profile, sides[side]["centerline"]
        ):
            ax.axvline(distance, color=color, linewidth=0.8, alpha=0.4)
            ax.text(distance, 0.5, str(number), color=color, fontsize=8, ha="center")
    ax.set_ylim(0, max_diameter * 1.4)
    ax.set_xlabel("Distance from aortic bifurcation [mm]", color="white")
    ax.set_ylabel("Diameter [mm]", color="white")
    ax.grid(axis="y", color="white", alpha=0.15, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.4)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, facecolor="black")
    plt.close(fig)
    return output_path


def _cpr_profile_y(profile, nr_slices, step_mm):
    """Map profile points onto the display y-axis of a CPR (distal end at the bottom)."""
    distances = profile["distances"]
    arc_length_from_distal_end = distances[0] - distances
    return (nr_slices - 1) * step_mm - arc_length_from_distal_end


def _cpr_excluded_regions(profile, nr_slices, step_mm):
    """Excluded profile regions with their extent on the CPR display y-axis (mm)."""
    if profile is None or not len(profile["distances"]):
        return []
    # same mapping as _cpr_profile_y
    offset = (nr_slices - 1) * step_mm - profile["distances"][0]
    return [
        {**region, "y_min": offset + region["start_mm"], "y_max": offset + region["end_mm"]}
        for region in profile.get("excluded", [])
    ]


def _exclusion_summary(regions):
    if not regions:
        return "No regions excluded from measurement."
    lines = [f"{region['start_mm']}-{region['end_mm']} mm: {region['reason']}" for region in regions]
    return "Not measured (distance from aortic bifurcation):\n" + "\n".join(lines)


def plot_iliac_cpr(cprs, profiles, landmarks, affine, output_path, metadata, tmp_dir, vmin=-700, vmax=1000):
    """CPR overview: one row per side with sagittal, coronal CPR and diameter profile.

    cprs: {side: (ct_cpr_img, mask_cpr_img, cpr_info, centerline)}
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13.5, 8.5 * len(cprs)), facecolor="black")
    grid = fig.add_gridspec(len(cprs), 3, width_ratios=[1.0, 1.0, 1.2], wspace=0.16, hspace=0.25)
    for row, (side, (ct_cpr, mask_cpr, cpr_info, centerline)) in enumerate(cprs.items()):
        image_data = ct_cpr.get_fdata()
        img_sagittal, img_coronal, mask_sagittal, mask_coronal = _get_cpr_longitudinal_views(
            image_data, mask_cpr.get_fdata() > 0.5
        )
        nr_slices = image_data.shape[2]
        step_mm = float(ct_cpr.header.get_zooms()[2])
        side_landmarks = {
            number: {"cl_idx": landmark["cl_idx"], "name": landmark["display_name"], "empty": landmark["empty"]}
            for number, landmark in landmarks.items()
            if landmark.get("cl_side") == side
        }
        profile = profiles.get(side)
        excluded = _cpr_excluded_regions(profile, nr_slices, step_mm)
        landmark_positions = _get_landmark_positions(
            side_landmarks,
            np.asarray([vertex.point for vertex in centerline]),
            affine,
            cpr_info["sample_positions_mm"],
            nr_slices,
            step_mm,
        )
        if tmp_dir is not None:
            excluded_rows = [(region["y_min"] / step_mm, region["y_max"] / step_mm) for region in excluded]
            for view, image, mask in (("sagittal", img_sagittal, mask_sagittal), ("coronal", img_coronal, mask_coronal)):
                save_cpr_view(Path(tmp_dir) / f"cpr_{side}_{view}.png", image, mask, vmin, vmax,
                              landmark_positions, excluded_rows, SIDE_COLORS[side])
        axes = [fig.add_subplot(grid[row, col]) for col in range(3)]
        for ax, title, image, mask in (
            (axes[0], f"{side.capitalize()} - Sagittal CPR", img_sagittal, mask_sagittal),
            (axes[1], f"{side.capitalize()} - Coronal CPR", img_coronal, mask_coronal),
        ):
            ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bicubic")
            _plot_mask_contours(ax, mask, color="#ff4d4d", smooth=35, linewidth=1.8)
            for landmark in landmark_positions:
                ax.axhline(landmark["slice_idx"], color="white", linewidth=0.8, alpha=0.25)
            for region in excluded:
                ax.axhspan(region["y_min"] / step_mm, region["y_max"] / step_mm, color=SIDE_COLORS[side],
                           alpha=0.15, linewidth=0)
            ax.set_title(title, color="white", fontsize=12, pad=10)
            ax.axis("off")

        max_curve_mm = 10.0
        for region in excluded:
            axes[2].axhspan(region["y_min"], region["y_max"], facecolor=SIDE_COLORS[side], alpha=0.15,
                            edgecolor=SIDE_COLORS[side], hatch="///", linewidth=0)
        if profile is not None and len(profile["distances"]):
            curve_y = _cpr_profile_y(profile, nr_slices, step_mm)
            axes[2].plot(profile["minor"], curve_y, color=SIDE_COLORS[side], linewidth=1.8,
                         label="Short axis")
            axes[2].plot(profile["major"], curve_y, color=SIDE_COLORS[side], linewidth=0.8,
                         alpha=0.5, linestyle="--", label="Long axis")
            if np.isfinite(profile["major"]).any():
                max_curve_mm = max(max_curve_mm, float(np.nanmax(profile["major"])))
            handles, _ = axes[2].get_legend_handles_labels()
            if excluded:
                handles.append(Patch(facecolor=SIDE_COLORS[side], alpha=0.3, hatch="///", label="Not measured"))
            axes[2].legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                           ncol=len(handles), fontsize=9)
            axes[2].text(0.0, -0.12, _exclusion_summary(excluded), transform=axes[2].transAxes,
                         color=colors["orange"] if excluded else "white", fontsize=8, ha="left", va="top")
        _style_diameter_profile(axes[2], (nr_slices - 1) * step_mm, max_curve_mm, landmark_positions)
        axes[2].set_ylabel("Along centerline [mm]", color="white")
        axes[2].set_title("Diameter Profile", color="white", fontsize=12, pad=10)

    fig.canvas.draw()
    height, width = fig.canvas.get_width_height()[::-1]
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    content = Image.fromarray(rgba[:, :, :3].copy())
    plt.close(fig)
    image = compose_report_image(content, metadata=metadata, report_title=REPORT_TITLE, tmp_dir=tmp_dir)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def vessel_image(side_data, aorta, affine):
    return nib.Nifti1Image(((side_data["iliac"] > 0) | (aorta > 0)).astype(np.uint8), affine)
