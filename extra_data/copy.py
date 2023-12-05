import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import h5py

from .utils import progress_bar

__all__ = ["copy_structure"]


def progress(processed, total, *, show=True):
    """Show progress information"""
    if not show:
        return

    pbar = progress_bar(processed, total)
    if sys.stderr.isatty():
        # "\x1b[2K": delete whole line, "\x1b[1A": move up cursor
        print("\x1b[2K\x1b[1A\x1b[2K\x1b[1A", file=sys.stderr)
    print(pbar, file=sys.stderr)


class Cloner:
    def __init__(self, input, output, *, run_data=False, control_data=False):
        self.run_data = run_data
        self.control_data = control_data
        self.visited = {}

        if output.file.mode == "r":
            raise ValueError("Output file must be writeable.")
        self.visit(input, output)

    @staticmethod
    def _copy_attrs(input, output):
        for key, value in input.attrs.items():
            output.attrs.create(key, value)

    def visit(self, obj, output):
        if obj.name != "/":
            link = obj.file.get(obj.name, getlink=True)
            if isinstance(link, h5py.SoftLink):
                # note this works only for SoftLinks. ExternalLink object's
                # name is not the name of the path, but the targeted file's path
                output[obj.name] = link
                return

        obj_id = h5py.h5o.get_info(obj.id).addr

        if obj_id in self.visited:
            # Hardlink to an object we've already seen
            output[obj.name] = output[self.visited[obj_id]]
            return

        self.visited[obj_id] = obj.name

        if isinstance(obj, h5py.Dataset):
            if (
                obj.name.startswith("/INSTRUMENT")
                or (obj.name.startswith("/CONTROL") and not self.control_data)
                or (obj.name.startswith("/RUN") and not self.run_data)
            ):
                output_obj = output.create_dataset_like(obj.name, obj)
            else:
                # note: consider using h5py.File.copy once a bug causing
                # segfault for dataset with attributes is fixed,
                # see: https://github.com/HDFGroup/hdf5/issues/2414
                output_obj = output.create_dataset_like(obj.name, obj, data=obj[()])
            self._copy_attrs(obj, output_obj)
        elif isinstance(obj, h5py.Group):
            if obj == obj.file:
                # root object
                output_obj = output["/"]
            else:
                output_obj = output.create_group(obj.name)
            self._copy_attrs(obj, output_obj)

            for name, child in obj.items():
                if child.file != obj.file:
                    # external link
                    output[f'{obj.name}/{name}'] = obj.get(name, getlink=True)
                else:
                    self.visit(child, output)
        else:
            # unknown type
            return


def copy_structure(
    input: Union[Path, str],
    output: Union[Path, str],
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
    if isinstance(input, str):
        input = Path(input)
    input = input.expanduser()

    if isinstance(output, str):
        output = Path(output)
    output = output.expanduser()

    if not output.is_dir():
        raise ValueError(f"The given output directory does not exist: {output}")

    if h5py.is_hdf5(input):
        if output == input.parent:
            raise ValueError("Input and output must be different directories.")
        Cloner(
            h5py.File(input),
            h5py.File(output / input.name, "w"),
            run_data=run_data,
            control_data=control_data,
        )
    elif input.is_dir():
        if output == input:
            raise ValueError("Input and output must be different directories.")
        # clone all hdf5 file present in the given directory
        h5files = [f for f in input.glob("*") if h5py.is_hdf5(f)]

        progress(0, len(h5files), show=term_progress)
        for n, file_ in enumerate(h5files, start=1):
            Cloner(
                h5py.File(file_),
                h5py.File(output / file_.name, "w"),
                run_data=run_data,
                control_data=control_data,
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
        help="Copy data present in the CONTROL group.",
    )

    args = ap.parse_args(argv)

    print(f"Cloning file(s) structure:\ninput: {args.input}\nOutput: {args.output}\n")
    copy_structure(
        args.input,
        args.output,
        run_data=args.copy_run_data,
        control_data=args.copy_control_data,
        term_progress=True,
    )
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
