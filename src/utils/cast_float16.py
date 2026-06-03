"""
Cast a directory of float32 activation .npy files to float16 in place.

Halves on-disk size (e.g. ~4.4 TB -> ~2.2 TB for a full 56,982-file layer)
while keeping the files directly memory-mappable by the SAE training loader.

Each file is rewritten atomically (write .tmp, then os.replace) so a killed
job never leaves a truncated/corrupt .npy. Already-float16 files are skipped,
so the job is safe to re-run / resume.

Usage:
  python -m src.utils.cast_float16 --dir /scratch/euh7ys/activations_layer08_train
  python -m src.utils.cast_float16 --dir <src_dir> --out_dir <dst_dir>   # copy-convert
"""
import os
import argparse
import glob
import time
from multiprocessing import Pool

import numpy as np


def convert_one(args):
    src_path, out_dir = args
    try:
        arr = np.load(src_path, mmap_mode="r")
        if arr.dtype == np.float16:
            return ("skipped", src_path, 0)

        out_path = src_path if out_dir is None else os.path.join(out_dir, os.path.basename(src_path))
        tmp_path = out_path + ".tmp"

        np.save(tmp_path, arr.astype(np.float16))
        os.replace(tmp_path, out_path)  # atomic
        return ("done", src_path, arr.size * 2)  # bytes freed ~ size*2 (4->2 bytes)
    except Exception as e:
        return (f"error: {e}", src_path, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Directory of .npy files to convert")
    p.add_argument("--out_dir", default=None,
                   help="If set, write converted files here instead of in place")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers (keep low when free disk space is tight)")
    p.add_argument("--glob", default="*.npy")
    args = p.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files:
        raise FileNotFoundError(f"No files matching {args.glob} in {args.dir}")

    print(f"Directory : {args.dir}")
    print(f"Files     : {len(files)}")
    print(f"Mode      : {'in-place' if args.out_dir is None else 'copy -> ' + args.out_dir}")
    print(f"Workers   : {args.workers}\n", flush=True)

    work = [(f, args.out_dir) for f in files]
    done = skipped = errors = 0
    freed = 0
    t0 = time.perf_counter()

    with Pool(args.workers) as pool:
        for i, (status, path, bytes_freed) in enumerate(pool.imap_unordered(convert_one, work, chunksize=8)):
            if status == "done":
                done += 1
                freed += bytes_freed
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"  {status}  {os.path.basename(path)}", flush=True)

            if (i + 1) % 1000 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(files)}] done={done} skipped={skipped} "
                      f"errors={errors} freed~{freed/1e12:.2f}TB ({rate:.0f} files/s)", flush=True)

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.0f}s — converted={done} skipped={skipped} "
          f"errors={errors} freed~{freed/1e12:.2f}TB")


if __name__ == "__main__":
    main()
