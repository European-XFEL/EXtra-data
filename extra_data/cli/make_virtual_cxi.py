import argparse
import logging
import os
import os.path as osp
import re
import sys
from textwrap import dedent

from extra_data import RunDirectory
from extra_data.components import identify_multimod_detectors

log = logging.getLogger(__name__)

def parse_number(number:str):
    try:
        return float(number)
    except ValueError:
        return int(number, 0)

def get_sequences (det_data):
    """
    Prepare a set of sequence names from the detector data in a run.

    Parameters
    ----------

    det_data: extra_data.components.MultimodDetectorBase
      The detector data instance for a run.

    Returns
    -------

    list
      Sorted list of strings of the file sequences names.
    """
    sequences = set()
    for source in det_data.source_to_modno:
        for file in det_data.data._source_index[source]:
            mobj = re.search(r'-(S\d+)\.h5', file.filename)
            if mobj is not None:
                sequences.add(mobj.group(1))
    return sorted(sequences)

def main(argv=None):
    example = dedent("""
        Example:

          extra-data-make-virtual-cxi -o ./out_file.h5 --min-modules 15 \\
            --fill-value data 0 --fill-value gain 1 /path/to/source/run
    """)
    ap = argparse.ArgumentParser(
        'extra-data-make-virtual-cxi', epilog=example,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Write a virtual CXI file to access the detector data.'
    )
    ap.add_argument('run_dir', help="Path to an EuXFEL run directory")
    # Specifying a proposal directory & a run number is the older interface.
    # If the run_number argument is passed, run_dir is used as proposal.
    ap.add_argument('run_number', nargs="?", help=argparse.SUPPRESS)
    ap.add_argument(
        '-o', '--output',
        help="Filename or path for the CXI output file. "
             "By default, it is written in the proposal's scratch directory."
    )
    ap.add_argument(
        '--min-modules', type=int, default=None, metavar='N',
        help='Include trains where at least N modules have data (default:'
             ' half+1 of all detector modules).'
    )
    ap.add_argument(
        '--n-modules', type=int, default=None, metavar='N',
        help='Number of detector modules in the experiment setup.'
             ' Should be used only for JUNGFRAU data.'
    )
    ap.add_argument(
        '--fill-value', action='append', nargs=2, metavar=('DS', 'V'),
        help='define fill value (V) for individual dataset (DS). Datasets are'
             ' "data", "gain" and "mask". (defaults: data: nan (proc, float32)'
             ' or 0 (raw, uint16); gain: 0; mask: 0xffffffff)'
    )
    ap.add_argument(
        '--exc-suspect-trains', action='store_true',
        help='Exclude suspect trains. This tries to avoid some issues with'
             ' incorrect train IDs in the data, but may mean less data is'
             ' available.'
    )
    ap.add_argument(
        '-seq', '--per-sequence', action='store_true',
        help='Create separate CXI files for each data sequence.'
    )
    args = ap.parse_args(argv)
    out_file = args.output
    fill_values = None
    if args.fill_value:
        fill_values = {ds: parse_number(value) for ds, value in args.fill_value}

    logging.basicConfig(level=logging.INFO)

    if args.run_number:
        # proposal directory, run number
        run  = 'r%04d' % int(args.run_number)
        proposal = args.run_dir
        run_dir = osp.join(args.run_dir, 'proc', run)
        if out_file is None:
            out_file = osp.join(proposal, 'scratch', '{}_detectors_virt.cxi'.format(run))

    else:
        # run directory
        run_dir = os.path.abspath(args.run_dir)
        if out_file is None:
            m = re.search(r'/(raw|proc)/(r\d{4})/?$', run_dir)
            if not m:
                sys.exit("ERROR: '-o outfile' option needed when "
                         "input directory doesn't look like .../proc/r0123")
            proposal = run_dir[:m.start()]
            fname = '{}_{}_detectors_virt.cxi'.format(*m.group(2, 1))
            out_file = osp.join(proposal, 'scratch', fname)

    out_dir = osp.dirname(osp.abspath(out_file))

    if not os.access(run_dir, os.R_OK):
        sys.exit("ERROR: Don't have read access to {}".format(run_dir))
    if not os.access(out_dir, os.W_OK):
        sys.exit("ERROR: Don't have write access to {}".format(out_dir))

    log.info("Reading run directory %s", run_dir)
    inc_suspect = not args.exc_suspect_trains
    run = RunDirectory(run_dir, inc_suspect_trains=inc_suspect)

    _, det_class = identify_multimod_detectors(run, single=True)

    min_modules = args.min_modules
    if min_modules is None:
        det_n_modules = getattr(det_class, 'n_modules', 0)
        min_modules = (det_n_modules // 2) + 1

    det = det_class(run, min_modules=min_modules, n_modules=args.n_modules)
    if not args.per_sequence:
        det.write_virtual_cxi(out_file, fill_values)
    else:
        for sequence in get_sequences(det):
            seq_wildcard = f"*{sequence}.h5"
            split_out = out_file.rsplit('.', 1)
            seq_out_file = f"{split_out[0]}_{sequence}.{split_out[1]}"

            seq_run = RunDirectory(
                run_dir, include=seq_wildcard, inc_suspect_trains=inc_suspect)
            seq_det = det_class(
                seq_run, min_modules=min_modules, n_modules=args.n_modules)
            seq_det.write_virtual_cxi(seq_out_file, fill_values)

if __name__ == '__main__':
    main()
