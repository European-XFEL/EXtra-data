import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py


def clone_file_structure(h5file: Path, output: Path) -> None:
    clone = h5py.File(output / h5file.name, "w")

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            clone.create_group(name)
        elif isinstance(obj, h5py.Dataset):
            if (
                name.startswith("INSTRUMENT")
                or name.startswith("CONTROL")
                or name.startswith("RUN")
            ):
                clone.create_dataset_like(obj)
            else:
                clone.create_dataset_like(obj, data=obj[()])

    original = h5py.File(h5file)
    original.visititems(visitor)


def clone(input: Path, output: Path) -> None:
    """Clone EuXFEL HDF5 file structure without any of its data.
    
    Clone the input file or files present the input directory.
    The cloned files will be writen to output.
    """
    if not output.is_dir():
        raise ValueError(f"The given output directory does not exist: {output}")

    if h5py.is_hdf5(input):
        if output == input.parent:
            raise ValueError("Input and output must be different directories.")
        clone_file_structure(input, output)
    elif input.is_dir():
        if output == input:
            raise ValueError("Input and output must be different directories.")
        # clone all hdf5 file present in the given directory
        for file_ in input.glob("*"):
            if not h5py.is_hdf5(file_):
                continue
            clone_file_structure(file_, output)
    else:
        raise ValueError(f"invalid input: {input}")


def main(argv=None):
    ap = ArgumentParser("Clone EuXFEL HDF5 files but with empty datasets.")
    ap.add_argument("input", type=str, help="Path to an HDF5 file or a directory.")
    ap.add_argument(
        "output", type=str, help="Output directory to write the cloned files."
    )

    args = ap.parse_args()

    path_in = Path(args.input).expanduser()
    path_out = Path(args.output).expanduser()

    clone(path_in, path_out)


if __name__ == "__main__":
    main(sys.argv[1:])
