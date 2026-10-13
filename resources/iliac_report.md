# Iliac artery report

`totalseg_iliac_report` creates a quantitative report of both iliac arteries from a CT image. It
places 7 landmarks on centerlines running from the lower abdominal aorta into each iliac artery,
measures the vessel at each landmark, summarizes each side, and renders the results as JSON and an
RGB NIfTI report. It can also generate a curved planar reformation (CPR) of both sides.

The report is a research prototype and is not intended for clinical diagnosis.
For research use, cite the
[TotalSegmentator paper](https://pubs.rsna.org/doi/10.1148/ryai.230024).

## Requirements

The report only uses the open `total` task, so it does not require a license.

The report renderer requires `wkhtmltopdf` and a virtual X server. On Ubuntu/Debian:

```bash
sudo apt-get install wkhtmltopdf xvfb
```

## Running the complete pipeline

Use `--run_models` to create all required masks from the CT:

```bash
totalseg_iliac_report \
  -i ct.nii.gz \
  -o iliac_report.nii.gz \
  -j iliac_report.json \
  -l iliac_report.log \
  -c iliac_cpr.png \
  --run_models
```

Patient metadata can optionally be supplied with `-n nodeinfo.json`. Without it, anonymous default
metadata is used.

Precomputed masks can be supplied instead of `--run_models`:

```bash
totalseg_iliac_report \
  -i ct.nii.gz \
  -rt masks/ \
  -o iliac_report.nii.gz \
  -j iliac_report.json \
  -l iliac_report.log
```

The ROI directory must contain `aorta.nii.gz`, `iliac_artery_left.nii.gz` and
`iliac_artery_right.nii.gz` (as produced by the `total` task).

## Segmentation pipeline

1. A fast `total` task (`--roi_subset aorta iliac_artery_left iliac_artery_right`) defines a crop.
2. The CT is cropped with a 20 mm margin around both iliac arteries and the aorta up to 60 mm above
   the iliac arteries.
3. The cropped CT is resampled to 0.8 mm isotropic spacing.
4. The normal `total` task segments the aorta and both iliac arteries.

## Centerlines

For each side, a centerline is extracted from the union of the lower aorta (up to 40 mm above the
iliac arteries) and the iliac artery. It runs from the distal end of the iliac artery to the aorta
and is smoothed to remove the voxel staircase of the skeleton. The **aortic bifurcation** of a side
is the point where its centerline leaves the aorta mask.

The iliac segmentation is frequently interrupted or narrowed at the transition from the common to
the external iliac artery. Fragments closer than 20 mm are connected for the centerline only;
smaller fragments (<100 mm³) and fragments further away are removed.

## Landmark definitions

1. **Aortic bifurcation** — 10 mm cranial of the bifurcation on the right centerline (left if the
   right side is missing), measured on the aorta.
2. / 5. **Right / left iliac proximal** — 10 mm distal of the bifurcation.
3. / 6. **Right / left iliac max.** — the position of the largest short-axis diameter between the
   proximal and distal landmark.
4. / 7. **Right / left iliac distal** — 10 mm proximal of the distal end of the iliac centerline.

For each available landmark, a plane perpendicular to the centerline is intersected with the vessel
mask. The JSON contains the cross-sectional area in cm², the maximum and the perpendicular diameter
in cm, and the diameter endpoints in voxel and world coordinates.

## Section measurements

For each side (`iliac_right`, `iliac_left`) the section runs from the aortic bifurcation to the
distal end of the iliac centerline:

- `length`: centerline length in cm;
- `volume`: volume of the iliac artery mask in ml;
- `max_diameter` / `min_diameter`: largest and smallest **short-axis** diameter (perpendicular to
  the long axis of the cross-section) along the centerline in cm. The short axis is used because
  branch origins (internal iliac artery) and oblique planes inflate the long-axis diameter;
- `tortuosity_index`: centerline length divided by the straight distance between its endpoints
  (1.0 = straight);
- `excluded_regions`: parts of the centerline that were not used for `max_diameter` /
  `min_diameter`, each with `reason`, `start_mm` and `end_mm` (distance from the bifurcation).

The diameter profile ignores the first and last 5 mm of each section. In addition the following
regions are excluded, reported in the JSON and the rendered report, and logged as a warning:

- **segmentation gap**: within 10 mm of a gap in the segmentation, where the mask tapers off;
- **focal narrowing**: within 10 mm of a position where the short-axis diameter drops below 50% of
  the median of the section. This is usually a segmentation artifact, but can also be a real
  stenosis, so these regions should be reviewed in the CPR.

## CPR and rendered report

The rendered report contains a rotating 3D view with the landmark planes, the cross-section of every
landmark, the short- and long-axis diameter profiles of both sides (excluded regions shaded) and the
section table including the excluded regions.

When `-c` is given, the CT is sampled on planes along each side's centerline. The CPR overview
contains a row per side with sagittal and coronal straightened views, vessel contours, landmark
positions and the diameter profile.

## JSON structure

The top-level JSON contains:

- `landmarks`: entries `1` through `7` with names and measurements;
- `section_stats`: measurements for `iliac_right` and `iliac_left`;
- `metadata`: normalized patient/study metadata and the report version.

Unavailable measurements are represented as `null`.

## Limitations

- The `iliac_artery_*` masks cover the common and external iliac artery, not the internal iliac
  artery. Landmarks therefore do not mark the common/external iliac transition.
- Report quality depends directly on the aorta and iliac artery segmentation. The masks do not
  distinguish lumen, thrombus and calcified wall.
- Missing anatomy or a restricted field of view (e.g. no aorta above the bifurcation) makes a side
  unavailable.
- CPR and 3D rendering are computationally and memory intensive.
- Input validation currently returns without an error status, so automation should verify that all
  requested output files were created.
