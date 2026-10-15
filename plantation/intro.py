"""Title, banner illustration and the assessment description shown at the top of the app."""
from __future__ import annotations

TITLE = "🌱 Where to Plant Next: Rainfall Readiness by District"
SUBTITLE = "Decadal precip assessment (JJASOND)"

INTRO = (
    "Young saplings live or die by the rain that falls in their first season. This screen ranks "
    "156 districts across Andhra Pradesh, Karnataka, Odisha, Tamil Nadu and Telangana by their "
    "average June–December rainfall over the last ten years (2016–2025), so you can shortlist "
    "the rain-ready districts before looking closer."
)

ABOUT_MD = """
**What it measures**

The 10-year average (2016–2025) of total rainfall, in mm, for each district. The default window is
**June to December**, the planting and establishment season (JJASOND). Switch to **Jan–Dec** for the
full-year total.

**How to use it**

Set the **threshold** in the sidebar (default **800 mm**). Districts whose 10-year average is at or above it are
*shortlisted* as rain-ready. Change the threshold to see how the shortlist grows or shrinks, and filter by
state or search for a district by name.

**How the number is made**

For every day of every year, rainfall is averaged over the grid cells inside the district (weighted by how much
of each cell lies in the district), added up for the window, and then averaged over the ten years. Two
independent sources are available: **IMD** (India Meteorological Department gauge-based grid, 0.25°, the
default) and **GSMaP** (JAXA satellite estimates, 0.1° grid).

**Reading the results**

Blue districts on the map are shortlisted; the deeper the blue, the wetter the district. Grey districts fall
below the threshold. The ranked table lists the same districts from wettest to driest, with a CSV download.

**Limits**

This is a first-pass screen, not a suitability score: soil, irrigation, species and slope are not considered.
The two sources can disagree, and a few districts sit on different sides of the threshold depending on which you
pick, so check both. IMD's coarser grid gives small districts only a few cells (flagged in the table's Notes).
Days missing at the source are skipped and noted.
"""


def banner_svg() -> str:
    """A colourful sunrise-over-plantation illustration, as a self-contained SVG string."""
    sky = ('<defs>'
           '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">'
           '<stop offset="0" stop-color="#1F8A9E"/><stop offset="0.55" stop-color="#7FD1C0"/>'
           '<stop offset="1" stop-color="#FFD98A"/></linearGradient>'
           '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
           '<stop offset="0" stop-color="#FFE9A8" stop-opacity="0.95"/>'
           '<stop offset="1" stop-color="#FFE9A8" stop-opacity="0"/></radialGradient>'
           '</defs>'
           '<rect width="1200" height="220" fill="url(#sky)"/>'
           '<circle cx="930" cy="120" r="120" fill="url(#glow)"/>'
           '<circle cx="930" cy="120" r="46" fill="#FFC93C"/>')

    clouds = ""
    for cx, cy, s in ((230, 52, 1.0), (560, 38, 0.8), (800, 62, 0.65), (1070, 40, 0.9)):
        clouds += (f'<g transform="translate({cx} {cy}) scale({s})" fill="#F4F7FA">'
                   '<ellipse cx="0" cy="0" rx="62" ry="22"/><ellipse cx="-34" cy="6" rx="36" ry="17"/>'
                   '<ellipse cx="38" cy="8" rx="40" ry="18"/><ellipse cx="6" cy="-14" rx="34" ry="18"/></g>')
        for i, dx in enumerate((-36, -12, 12, 36)):                # rain falling under each cloud
            x, y = cx + dx * s, cy + 30 * s
            clouds += (f'<path d="M{x:.0f} {y + i % 2 * 8:.0f} l-6 22" stroke="#2E86DE" '
                       f'stroke-width="3" stroke-linecap="round"/>')

    hills = ('<path d="M0 150 C180 110 330 170 520 138 C720 104 900 168 1200 126 L1200 220 L0 220 Z" fill="#8BC34A"/>'
             '<path d="M0 172 C220 140 420 196 640 164 C860 134 1010 190 1200 160 L1200 220 L0 220 Z" fill="#4CAF50"/>'
             '<path d="M0 196 C260 176 520 214 760 190 C960 170 1090 206 1200 188 L1200 220 L0 220 Z" fill="#2E7D32"/>')

    trees = ""
    greens = ("#1B5E20", "#2E7D32", "#388E3C", "#66BB6A")
    for row, (y, scale) in enumerate(((176, 0.55), (190, 0.75), (206, 1.0))):
        for k in range(18 if row == 0 else 14 if row == 1 else 10):
            x = 40 + k * (1120 / (17 if row == 0 else 13 if row == 1 else 9)) + (25 if row == 1 else 0) * (k % 2)
            g = greens[(k + row) % len(greens)]
            trees += (f'<g transform="translate({x:.0f} {y}) scale({scale})">'
                      f'<rect x="-2" y="-14" width="4" height="16" fill="#6D4C41"/>'
                      f'<circle cx="0" cy="-22" r="11" fill="{g}"/>'
                      f'<circle cx="-7" cy="-16" r="8" fill="{greens[(k + row + 1) % len(greens)]}"/>'
                      f'<circle cx="7" cy="-16" r="8" fill="{greens[(k + row + 2) % len(greens)]}"/></g>')

    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 220" '
            'preserveAspectRatio="xMidYMid slice" role="img" '
            'aria-label="Sunrise over rows of young trees under rain clouds">'
            + sky + clouds + hills + trees + "</svg>")
