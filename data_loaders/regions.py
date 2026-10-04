"""
Region definitions for atmospheric river analysis.

Each region specifies lat/lon bounding boxes and winter months
(when ARs are most impactful). Used for filtering AR masks and
computing per-region SAE feature statistics.
"""

# Regions used in AR validation analysis.
# Keys match those used in the original Layer8VSAE_AR.ipynb notebook.
#
# lon uses 0–360 convention (matching ERA5 / AR mask coordinates).
# Western Europe wraps the prime meridian, so it has two lon ranges.
AR_REGIONS = {
    "Western US": {
        "lat_min": 32, "lat_max": 48,
        "lon_min": 235, "lon_max": 245,
        "winter_months": [11, 12, 1, 2, 3],
    },
    "Western Europe": {
        "lat_min": 40, "lat_max": 58,
        "lon_min": 350, "lon_max": 360,
        "lon_min2": 0, "lon_max2": 10,
        "winter_months": [10, 11, 12, 1, 2, 3],
    },
    "Western South America": {
        "lat_min": -45, "lat_max": -30,
        "lon_min": 285, "lon_max": 295,
        "winter_months": [5, 6, 7, 8, 9],
    },
    "Eastern Australia": {
        "lat_min": -45, "lat_max": -25,
        "lon_min": 145, "lon_max": 160,
        "winter_months": [5, 6, 7, 8, 9],
    },
}
