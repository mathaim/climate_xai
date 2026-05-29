#!/usr/bin/env python3
"""
Atmospheric river (AR) ground-truth label loading.

Loads AR event masks from a NetCDF file produced by an external AR
detection algorithm and classifies timesteps as AR / non-AR for
each analysis region.

The AR mask file contains:
  - Variable: ``event_masks`` — integer mask (0 = no AR, nonzero = AR event ID)
  - Dimensions: (time, lat, lon)
  - Time: integer steps from a base date, each step = 6 hours

Usage::

    from src.data.ar_labels import ARLabels

    ar = ARLabels("data/ar_labels/ar_masks.nc")
    print(ar)                            # summary of dates and AR counts

    # Get AR/non-AR dates for a specific region (winter only)
    ar_dates, no_ar_dates = ar.get_region_dates("Western US")

    # Check whether a single date is an AR event in a region
    ar.is_ar("Western US", pd.Timestamp("2020-01-15"))

    # Get all region classifications at once
    all_regions = ar.get_all_region_dates()
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Optional, Set, Tuple

from src.data.regions import AR_REGIONS


class ARLabels:
    """Load and query atmospheric river ground-truth labels.

    Parameters
    ----------
    ar_mask_path : str
        Path to the AR masks NetCDF file (``ar_masks.nc``).
    base_date : str or pd.Timestamp
        Reference date for the time coordinate. Each time value ``t``
        maps to ``base_date + t * 6 hours``.  Default: "2020-01-01".
    regions : dict, optional
        Region definitions. Defaults to :data:`AR_REGIONS`.
    """

    def __init__(
        self,
        ar_mask_path: str,
        base_date: str = "2020-01-01",
        regions: Optional[dict] = None,
    ):
        self.ar_mask_path = ar_mask_path
        self.base_date = pd.Timestamp(base_date)
        self.regions = regions or AR_REGIONS

        # Load dataset
        self._ds = xr.open_dataset(ar_mask_path, engine="netcdf4")
        self._event_masks = self._ds["event_masks"]

        # Build date arrays
        self.dates = np.array([
            self.base_date + pd.Timedelta(hours=int(t) * 6)
            for t in self._ds.time.values
        ])
        self.months = np.array([d.month for d in self.dates])
        self.n_timesteps = len(self.dates)

        # Cache region classifications
        self._region_cache: Dict[str, Tuple[Set, Set]] = {}

    def __repr__(self) -> str:
        return (
            f"ARLabels(path={self.ar_mask_path!r}, "
            f"n_timesteps={self.n_timesteps}, "
            f"date_range={self.dates[0]}..{self.dates[-1]}, "
            f"regions={list(self.regions.keys())})"
        )

    def _classify_region(self, region_name: str) -> Tuple[Set, Set]:
        """Classify timesteps as AR/non-AR for a region (winter only)."""
        if region_name in self._region_cache:
            return self._region_cache[region_name]

        cfg = self.regions[region_name]

        # Select AR presence in region bounding box
        if "lon_min2" in cfg:
            # Region wraps the prime meridian (e.g., Western Europe)
            ar_r1 = self._event_masks.sel(
                lat=slice(cfg["lat_min"], cfg["lat_max"]),
                lon=slice(cfg["lon_min"], cfg["lon_max"]),
            )
            ar_r2 = self._event_masks.sel(
                lat=slice(cfg["lat_min"], cfg["lat_max"]),
                lon=slice(cfg["lon_min2"], cfg["lon_max2"]),
            )
            ar_present = (
                (ar_r1 != 0).any(dim=["lat", "lon"])
                | (ar_r2 != 0).any(dim=["lat", "lon"])
            )
        else:
            ar_region = self._event_masks.sel(
                lat=slice(cfg["lat_min"], cfg["lat_max"]),
                lon=slice(cfg["lon_min"], cfg["lon_max"]),
            )
            ar_present = (ar_region != 0).any(dim=["lat", "lon"])

        # Filter to winter months
        winter_mask = np.isin(self.months, cfg["winter_months"])
        ar_winter = ar_present.values & winter_mask
        no_ar_winter = ~ar_present.values & winter_mask

        ar_dates = set(self.dates[ar_winter])
        no_ar_dates = set(self.dates[no_ar_winter])

        self._region_cache[region_name] = (ar_dates, no_ar_dates)
        return ar_dates, no_ar_dates

    def get_region_dates(
        self, region_name: str
    ) -> Tuple[Set[pd.Timestamp], Set[pd.Timestamp]]:
        """Get AR and non-AR winter dates for a region.

        Parameters
        ----------
        region_name : str
            Must be a key in ``self.regions``.

        Returns
        -------
        ar_dates : set of pd.Timestamp
            Winter timesteps where an AR was detected in the region.
        no_ar_dates : set of pd.Timestamp
            Winter timesteps with no AR in the region.
        """
        if region_name not in self.regions:
            raise ValueError(
                f"Unknown region {region_name!r}. "
                f"Available: {list(self.regions.keys())}"
            )
        return self._classify_region(region_name)

    def get_all_region_dates(
        self,
    ) -> Dict[str, Tuple[Set[pd.Timestamp], Set[pd.Timestamp]]]:
        """Get AR/non-AR dates for all regions.

        Returns
        -------
        dict
            ``{region_name: (ar_dates, no_ar_dates)}`` for each region.
        """
        return {name: self.get_region_dates(name) for name in self.regions}

    def is_ar(self, region_name: str, date: pd.Timestamp) -> bool:
        """Check if a specific date is an AR event in a region."""
        ar_dates, _ = self.get_region_dates(region_name)
        return date in ar_dates

    def summary(self) -> str:
        """Print a summary of AR counts per region."""
        lines = [f"AR Labels: {self.n_timesteps} timesteps"]
        lines.append(f"  Date range: {self.dates[0]} to {self.dates[-1]}")
        lines.append(f"  Base date: {self.base_date}")
        lines.append("")
        for name in self.regions:
            ar_dates, no_ar_dates = self.get_region_dates(name)
            lines.append(
                f"  {name}: {len(ar_dates)} AR / "
                f"{len(no_ar_dates)} non-AR winter timesteps"
            )
        return "\n".join(lines)
