from shutil import copytree

from testpath import assert_isfile

from extra_data import RunDirectory, voview

def test_main(mock_spb_raw_run, tmp_path):
    voview_file = tmp_path / 'run_overview.h5'
    voview.main([mock_spb_raw_run, '--overview-file', str(voview_file)])

    assert_isfile(voview_file)

    res = voview.main([mock_spb_raw_run, '--overview-file', str(voview_file), '--check'])
    assert res in (0, None)


def test_use_voview(mock_spb_raw_run, tmp_path):
    new_run_dir = tmp_path / 'r0238'
    copytree(mock_spb_raw_run, new_run_dir)
    voview_file = new_run_dir / 'overview.h5'

    run_orig = RunDirectory(str(new_run_dir), _use_voview=False)
    assert len(run_orig.files) > 1

    assert voview.find_file_write(new_run_dir) == str(voview_file)
    vofw = voview.VirtualOverviewFileWriter(voview_file, run_orig)
    vofw.write()

    run = RunDirectory(str(new_run_dir))
    assert [f.filename for f in run.files] == [str(voview_file)]
    assert len(run.train_ids) == 64

    assert 'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf' in run.instrument_sources
    assert 'SA1_XTD2_XGM/DOOCS/MAIN' in run.control_sources
