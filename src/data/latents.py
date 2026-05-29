#!/usr/bin/env python3
"""
SAE latent file loading — supports both dense (.npy) and sparse (.npz) formats.

Dense format: ``(n_nodes, n_latents)`` float32 — full latent vectors
Sparse format: ``indices`` (n_nodes, k) int16 + ``values`` (n_nodes, k) float32

Usage::

    from src.data.latents import LatentLoader

    # Dense latents
    loader = LatentLoader("results/matryoshka_latents_layer08", prefix="matryoshka_encoded_t")
    for ts, latents in loader:
        print(ts, latents.shape)  # e.g. "2020-01-01T00-00", (40962, 4096)

    # Sparse latents
    loader = LatentLoader("results/sparse_latents", prefix="sae_sparse_t", sparse=True)
    for ts, indices, values in loader:
        print(ts, indices.shape, values.shape)  # (40962, 32), (40962, 32)

    # Load single timestep
    latents = loader.load("2020-01-15T00-00")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Dict, Iterator, List, Optional, Tuple, Union


class LatentLoader:
    """Load SAE latent activations from disk.

    Parameters
    ----------
    latent_dir : str
        Directory containing latent files.
    prefix : str
        Filename prefix before the timestamp.
        E.g. ``"matryoshka_encoded_t"`` or ``"sae_sparse_t"``.
    sparse : bool
        If True, expect ``.npz`` files with ``indices`` and ``values``
        arrays. If False (default), expect dense ``.npy`` files.
    """

    def __init__(
        self,
        latent_dir: str,
        prefix: str = "matryoshka_encoded_t",
        sparse: bool = False,
    ):
        self.latent_dir = Path(latent_dir)
        self.prefix = prefix
        self.sparse = sparse
        self._ext = ".npz" if sparse else ".npy"

        self._files = sorted(
            glob(str(self.latent_dir / f"{prefix}*{self._ext}"))
        )
        assert len(self._files) > 0, (
            f"No {self._ext} files with prefix {prefix!r} in {latent_dir}"
        )

        # Build timestamp → filepath mapping
        self._ts_to_file: Dict[str, str] = {}
        for f in self._files:
            stem = Path(f).stem
            ts = stem.replace(prefix, "")
            self._ts_to_file[ts] = f

    def __repr__(self) -> str:
        fmt = "sparse" if self.sparse else "dense"
        return (
            f"LatentLoader(dir={self.latent_dir}, "
            f"prefix={self.prefix!r}, "
            f"format={fmt}, "
            f"n_files={len(self._files)})"
        )

    def __len__(self) -> int:
        return len(self._files)

    @property
    def timestamps(self) -> List[str]:
        """All available timestamps as strings."""
        return sorted(self._ts_to_file.keys())

    def ts_to_date(self, ts: str) -> pd.Timestamp:
        """Convert a timestamp string to a pandas Timestamp.

        Handles the format used in filenames, e.g.
        ``"2020-01-15T00-00"`` → ``Timestamp('2020-01-15 00:00')``.
        """
        # Replace date-time separators to standard format
        clean = ts.replace("T", " ").replace("-", ":", 2)
        # First two colons are date separators, need to be dashes
        parts = clean.split(" ")
        if len(parts) == 2:
            date_part = parts[0].replace(":", "-", 2)
            return pd.Timestamp(f"{date_part} {parts[1]}")
        return pd.Timestamp(clean)

    def date_to_file(self) -> Dict[pd.Timestamp, str]:
        """Build a mapping from pd.Timestamp to filepath."""
        return {self.ts_to_date(ts): path for ts, path in self._ts_to_file.items()}

    def load(self, timestamp: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Load latents for a single timestep.

        Parameters
        ----------
        timestamp : str
            Timestamp string matching the filename.

        Returns
        -------
        np.ndarray
            Dense: shape ``(n_nodes, n_latents)``.
        tuple of np.ndarray
            Sparse: ``(indices, values)`` each shape ``(n_nodes, k)``.
        """
        if timestamp not in self._ts_to_file:
            raise KeyError(f"Timestamp {timestamp!r} not found.")

        path = self._ts_to_file[timestamp]
        if self.sparse:
            data = np.load(path)
            return data["indices"], data["values"]
        else:
            return np.load(path)

    def __iter__(self) -> Iterator:
        """Iterate over all timesteps.

        Yields
        ------
        For dense: ``(timestamp_str, latent_array)``
        For sparse: ``(timestamp_str, indices, values)``
        """
        for ts in sorted(self._ts_to_file.keys()):
            path = self._ts_to_file[ts]
            if self.sparse:
                data = np.load(path)
                yield ts, data["indices"], data["values"]
            else:
                yield ts, np.load(path)

    def iter_dense(self, mmap: bool = True) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate dense latent files with optional memory mapping.

        Parameters
        ----------
        mmap : bool
            If True, use ``mmap_mode='r'`` for lower memory usage.

        Yields
        ------
        timestamp : str
        latents : np.ndarray, shape (n_nodes, n_latents)
        """
        assert not self.sparse, "Use __iter__ for sparse files"
        mode = "r" if mmap else None
        for ts in sorted(self._ts_to_file.keys()):
            yield ts, np.load(self._ts_to_file[ts], mmap_mode=mode)
