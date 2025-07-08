import os
import shutil
from multiprocessing import Process
from pathlib import Path
from textwrap import dedent
from unittest import mock
from warnings import catch_warnings

import numpy as np
import pytest

from extra_data import open_run
from extra_data.reader import DEFAULT_ALIASES_FILE


def test_open_run(mock_spb_raw_and_proc_run):
    mock_data_root, raw_run_dir, proc_run_dir = mock_spb_raw_and_proc_run

    with mock.patch('extra_data.read_machinery.DATA_ROOT_DIR', mock_data_root):
        # With integers
        run = open_run(proposal=2012, run=238)
        paths = {f.filename for f in run.files}

        assert paths
        for path in paths:
            assert '/raw/' in path

        # With strings
        run = open_run(proposal='2012', run='238')
        assert {f.filename for f in run.files} == paths

        # With numpy integers
        run = open_run(proposal=np.int64(2012), run=np.uint16(238))
        assert {f.filename for f in run.files} == paths

        # With a proposal path
        prop_path = Path(mock_data_root, 'SPB', '201830', 'p002012')
        run = open_run(proposal=prop_path, run=238)
        assert {f.filename for f in run.files} == paths

        # Proc folder
        proc_run = open_run(proposal=2012, run=238, data='proc')

        proc_paths = {f.filename for f in proc_run.files}
        assert proc_paths
        for path in proc_paths:
            assert '/raw/' not in path

        # Helper function to write an alias file at a specific path
        def write_aliases(path):
            aliases_path.parent.mkdir(parents=True, exist_ok=True)
            aliases_path.write_text(dedent("""
            xgm: SA1_XTD2_XGM/DOOCS/MAIN
            """))

        # To set the aliases, we should be able to use a string relative to the
        # proposal directory.
        aliases_path = Path(mock_data_root) / "SPB/201830/p002012/foo.yml"
        write_aliases(aliases_path)
        run = open_run(2012, 238, data="all", aliases="{}/foo.yml")
        assert "xgm" in run.alias

        # And a proper path
        aliases_path = Path(mock_data_root) / "foo.yml"
        write_aliases(aliases_path)
        run = open_run(2012, 238, aliases=aliases_path)
        assert "xgm" in run.alias

        # And a plain string
        run = open_run(2012, 238, aliases=str(aliases_path))
        assert "xgm" in run.alias

        # If the default file exists, it should be used automatically
        aliases_path = Path(DEFAULT_ALIASES_FILE.format(mock_data_root + "/SPB/201830/p002012"))
        write_aliases(aliases_path)
        run = open_run(2012, 238)
        assert "xgm" in run.alias

        # Check that aliases are loaded for old proposals where proc contains
        # all sources from raw too. Necessary because the aliases are only
        # loaded once for the raw data but the proc DataCollection will be used
        # if all sources exist in proc.
        shutil.rmtree(proc_run_dir)
        shutil.copytree(raw_run_dir, proc_run_dir)
        run = open_run(2012, 238, data="all")
        assert "xgm" in run.alias


@pytest.mark.parametrize('location', ['all', ['raw', 'proc']],
                         ids=['all', 'list'])
def test_open_run_multiple(mock_spb_raw_and_proc_run, location):
    mock_data_root, raw_run_dir, proc_run_dir = mock_spb_raw_and_proc_run

    with mock.patch('extra_data.read_machinery.DATA_ROOT_DIR', mock_data_root):
        # Separate folders
        raw_run = open_run(proposal=2012, run=238, data='raw')
        proc_run = open_run(proposal=2012, run=238, data='proc')

        # All folders
        all_run = open_run(proposal=2012, run=238, data=location)

        # Raw contains all sources.
        assert raw_run.all_sources == all_run.all_sources

        # Proc is a true subset.
        assert proc_run.all_sources < all_run.all_sources

        for source, srcdata in all_run._sources_data.items():
            for file in srcdata.files:
                if '/DET/' in source:
                    # AGIPD data is in proc.
                    assert '/raw/' not in file.filename
                else:
                    # Non-AGIPD data is in raw.
                    # (CAM, XGM)
                    assert '/proc/' not in file.filename

        # Delete the proc data
        shutil.rmtree(proc_run_dir)
        assert not os.path.isdir(proc_run_dir)

        with catch_warnings(record=True) as w:
            # Opening a run with 'all', with no proc data
            all_run = open_run(proposal=2012, run=238, data=location)

            # Attempting to open the proc data should raise a warning
            assert len(w) == 1

        # It should have opened at least the raw data
        assert raw_run.all_sources == all_run.all_sources

        # Run that doesn't exist
        with pytest.raises(Exception):
            open_run(proposal=2012, run=999)

        # run directory exists but contains no data
        os.makedirs(proc_run_dir)
        with catch_warnings(record=True) as w:
            open_run(proposal=2012, run=238, data=location)
            assert len(w) == 1


def test_open_run_default(mock_spb_raw_and_modern_proc_run):
    mock_data_root, raw_run_dir, proc_run_dir = mock_spb_raw_and_modern_proc_run

    with mock.patch('extra_data.read_machinery.DATA_ROOT_DIR', mock_data_root):
        run = open_run(proposal=2012, run=238, data='default')

        # /DET/ names should come from raw data
        det_sources = {f'SPB_DET_AGIPD1M-1/DET/{m}CH0:xtdf' for m in range(16)}
        for s in det_sources:
            assert 'image.gain' not in run[s]
            for file in run[s].files:
                assert '/raw/' in file.filename

        # /CORR/ names should come from corrected data
        corr_sources = {f'SPB_DET_AGIPD1M-1/CORR/{m}CH0:output' for m in range(16)}
        for s in corr_sources:
            assert 'image.gain' in run[s]
            for file in run[s].files:
                assert '/proc/' in file.filename

        assert run.legacy_sources == {}

def open_run_daemonized_helper(mock_data_root):
    with mock.patch('extra_data.read_machinery.DATA_ROOT_DIR', mock_data_root):
        open_run(2012, 238, data="all", parallelize=False)

def test_open_run_daemonized(mock_spb_raw_and_proc_run):
    mock_data_root, raw_run_dir, proc_run_dir = mock_spb_raw_and_proc_run

    # Daemon processes can't start their own children, check that opening a run is still possible.
    p = Process(target=open_run_daemonized_helper, args=(mock_data_root,), daemon=True)
    p.start()
    p.join()

    assert p.exitcode == 0
