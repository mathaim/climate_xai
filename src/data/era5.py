#!/usr/bin/env python3
"""
ERA5 input data loading for GraphCast.

ERA5 reanalysis data is the input that feeds GraphCast. Each file
contains the atmospheric state at one timestep, regridded to the
GraphCast icosahedral mesh (40,962 nodes × N features).

This module loads ERA5 mesh data for use in feature interpretation:
comparing atmospheric conditions where SAE features fire vs. don't.

Usage::

    from src.data.era5 import ERA5MeshDataset

    era5 = ERA5MeshDataset(
        data_dir="data/era5/era5_mesh_2020_2021",
        feature_names_path="data/era5/feature_names.txt",
    )
    print(era5)                     # summary
    print(era5.feature_names[:5])   # e.g. ['temperature_850', ...]

    # Load a single timestep
    data = era5.load_timestep("2020-01-15T00")  # (40962, n_features)

    # Iterate matched pairs with SAE latents
    for ts, era5_data, latent_data in era5.iter_matched(latent_dir, prefix):
        ...
"""

import numpy as np
from pathlib import Path
from glob import glob
from typing import Iterator, List, Optional, Tuple


class ERA5MeshDataset:
    """Load ERA5 atmospheric data on the GraphCast mesh.

    Parameters
    ----------
    data_dir : str
        Directory with ``era5_inputs_*.npy`` files.
        Each file is shape ``(n_nodes, n_features)``.
    feature_names_path : str, optional
        Path to a text file with one feature name per line.
        If not provided, features are numbered.
    """

    def __init__(
        self,
        data_dir: str,
        feature_names_path: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self._files = sorted(glob(str(self.data_dir / "era5_inputs_*.npy")))
        assert len(self._files) > 0, f"No era5_inputs_*.npy in {data_dir}"

        # Parse timestamps from filenames
        self._ts_to_file = {}
        for f in self._files:
            ts = Path(f).stem.replace("era5_inputs_", "")
            self._ts_to_file[ts] = f

        # Load feature names
        self.feature_names: List[str] = []
        if feature_names_path and Path(feature_names_path).exists():
            with open(feature_names_path) as f:
                self.feature_names = [line.strip() for line in f if line.strip()]

        # Peek at shape
        sample = np.load(self._files[0], mmap_mode="r")
        self.n_nodes = sample.shape[0]
        self.n_features = sample.shape[1] if sample.ndim > 1 else 1

        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def __repr__(self) -> str:
        return (
            f"ERA5MeshDataset(dir={self.data_dir}, "
            f"n_files={len(self._files)}, "
            f"n_nodes={self.n_nodes}, "
            f"n_features={self.n_features})"
        )

    def __len__(self) -> int:
        return len(self._files)

    @property
    def timestamps(self) -> List[str]:
        """All available timestamps as strings."""
        return sorted(self._ts_to_file.keys())

    def load_timestep(self, timestamp: str) -> np.ndarray:
        """Load ERA5 data for a single timestep.

        Parameters
        ----------
        timestamp : str
            Timestamp string matching the filename, e.g. ``"2020-01-15T00"``.

        Returns
        -------
        np.ndarray
            Shape ``(n_nodes, n_features)``, float32.
        """
        if timestamp not in self._ts_to_file:
            raise KeyError(
                f"Timestamp {timestamp!r} not found. "
                f"Have {len(self._files)} files."
            )
        data = np.load(self._ts_to_file[timestamp]).astype(np.float32)
        if data.ndim == 1:
            data = data[:, None]
        return data

    def load_file(self, path: str) -> np.ndarray:
        """Load ERA5 data from a specific file path.

        Returns
        -------
        np.ndarray
            Shape ``(n_nodes, n_features)``, float32.
        """
        data = np.load(path).astype(np.float32)
        if data.ndim == 1:
            data = data[:, None]
        return data

    def iter_matched(
        self,
        latent_dir: str,
        latent_prefix: str = "sae_encoded_t",
        latent_ext: str = ".npy",
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        """Iterate over (timestamp, era5_data, latent_data) pairs.

        Matches ERA5 files with SAE latent files by timestamp.

        Parameters
        ----------
        latent_dir : str
            Directory containing SAE latent files.
        latent_prefix : str
            Filename prefix before the timestamp.
        latent_ext : str
            File extension for latent files (.npy or .npz).

        Yields
        ------
        timestamp : str
        era5_data : np.ndarray, shape (n_nodes, n_features)
        latent_data : np.ndarray
            Dense latent array if .npy, or dict-like if .npz.
        """
        latent_path = Path(latent_dir)
        latent_files = sorted(glob(str(latent_path / f"{latent_prefix}*{latent_ext}")))

        latent_ts_map = {}
        for f in latent_files:
            stem = Path(f).stem
            ts = stem.replace(latent_prefix, "")
            latent_ts_map[ts] = f

        # Find matching timestamps
        common_ts = sorted(set(self._ts_to_file.keys()) & set(latent_ts_map.keys()))

        for ts in common_ts:
            era5_data = self.load_file(self._ts_to_file[ts])
            if latent_ext == ".npz":
                latent_data = np.load(latent_ts_map[ts])
            else:
                latent_data = np.load(latent_ts_map[ts])
            yield ts, era5_data, latent_data
