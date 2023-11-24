import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py

from ..utils import progress_bar

__all__ = ["clone"]


def progress(processed, total, *, show=True):
    """Show progress information"""
    if not show:
        return

    pbar = progress_bar(processed, total)
    if sys.stderr.isatty():
        # "\x1b[2K": delete whole line, "\x1b[1A": move up cursor
        print("\x1b[2K\x1b[1A\x1b[2K\x1b[1A", file=sys.stderr)
    print(pbar, file=sys.stderr)


def _clone_file_structure(
    h5file: Path, output: Path, *, run_data=False, control_data=False
) -> None:
    clone = h5py.File(output / h5file.name, "w")

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            clone.create_group(name)
        elif isinstance(obj, h5py.Dataset):
            if (
                name.startswith("INSTRUMENT")
                or (name.startswith("CONTROL") and not control_data)
                or (name.startswith("RUN") and not run_data)
            ):
                clone.create_dataset_like(name, obj)
            else:
                clone.create_dataset_like(name, obj, data=obj[()])

    original = h5py.File(h5file)
    original.visititems(visitor)


def clone(
    input: Path,
    output: Path,
    *,
    run_data=False,
    control_data=False,
    term_progress=False,
) -> None:
    """Clone EuXFEL HDF5 file structure without any of its data.

    Clone the input file or files present the input directory.
    The cloned files will be written to output.

    args:
        run_data: Copy data in RUN group if set to True
        control_data: Copy data in CONTROL group if set to True
        term_progress: show progress in terminal if set to True
    """
    if not output.is_dir():
        raise ValueError(f"The given output directory does not exist: {output}")

    if h5py.is_hdf5(input):
        if output == input.parent:
            raise ValueError("Input and output must be different directories.")
        _clone_file_structure(
            input, output, run_data=run_data, control_data=control_data
        )
    elif input.is_dir():
        if output == input:
            raise ValueError("Input and output must be different directories.")
        # clone all hdf5 file present in the given directory
        h5files = [f for f in input.glob("*") if h5py.is_hdf5(f)]

        progress(0, len(h5files), show=term_progress)
        for n, file_ in enumerate(h5files, start=1):
            _clone_file_structure(
                file_, output, run_data=run_data, control_data=control_data
            )
            progress(n, len(h5files), show=term_progress)
    else:
        raise ValueError(f"invalid input: {input}")


def main(argv=None):
    ap = ArgumentParser("Clone EuXFEL HDF5 files but with empty datasets.")
    ap.add_argument("input", type=str, help="Path to an HDF5 file or a directory.")
    ap.add_argument(
        "output", type=str, help="Output directory to write the cloned files."
    )
    ap.add_argument(
        "--copy-run-data",
        "-cr",
        action="store_true",
        default=False,
        help="Copy data present in the RUN group.",
    )
    ap.add_argument(
        "--copy-control-data",
        "-cc",
        action="store_true",
        default=False,
        help="Copy dara present in the CONTROL group.",
    )

    args = ap.parse_args()

    path_in = Path(args.input).expanduser()
    path_out = Path(args.output).expanduser()

    print(f"Cloning file(s) structure:\ninput: {path_in}\nOutput: {path_out}\n")
    clone(
        path_in,
        path_out,
        run_data=args.copy_run_data,
        control_data=args.copy_control_data,
        term_progress=True,
    )
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
