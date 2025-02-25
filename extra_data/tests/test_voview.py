from pathlib import Path
from shutil import copytree

from testpath import assert_isfile

from extra_data import H5File, RunDirectory, voview

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

    run_metadata = run.run_metadata()
    run_orig_metadata = run_orig.run_metadata()

    # Check the format version specifically, this is the only metadata that may
    # differ from the rest.
    if run_orig_metadata['dataFormatVersion'] != "0.5":
        # For all versions above 0.5 (i.e. 1.0 onwards), we write the voview
        # files in the 1.0 format.
        assert run_metadata['dataFormatVersion'] == "1.0"

        del run_metadata['dataFormatVersion']
        del run_orig_metadata['dataFormatVersion']
    else:
        assert run_metadata['dataFormatVersion'] == \
            run_orig_metadata['dataFormatVersion']

    # Check the rest of the metadata
    assert run_metadata == run_orig_metadata

    assert 'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf' in run.instrument_sources
    assert 'SA1_XTD2_XGM/DOOCS/MAIN' in run.control_sources

    with RunDirectory(str(new_run_dir)) as run:
        assert 'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf' in run.instrument_sources
        assert 'SA1_XTD2_XGM/DOOCS/MAIN' in run.control_sources

    xgm_intens = run['SA1_XTD2_XGM/DOOCS/MAIN:output', 'data.intensityTD']
    assert {p.name for p in xgm_intens.source_file_paths} == {
        'RAW-R0238-DA01-S00000.h5', 'RAW-R0238-DA01-S00001.h5'
    }
    assert {p.name for p in xgm_intens[:30].source_file_paths} == {
        'RAW-R0238-DA01-S00000.h5'
    }
    assert {p.name for p in xgm_intens[:0].source_file_paths} == {
        'RAW-R0238-DA01-S00000.h5'
    }
    assert xgm_intens.units == 'μJ'
    assert xgm_intens.units_name == 'microjoule'

    xgm_src = run['SA1_XTD2_XGM/DOOCS/MAIN:output']
    src_grp = xgm_src.files[0].file[f'INSTRUMENT/{xgm_src.source}']
    assert src_grp.attrs['source_files'][:].tolist() == [
        str(new_run_dir / f'RAW-R0238-DA01-S{i:05}.h5') for i in range(2)
    ]


def test_make_voview_missing_data(mock_fxe_raw_run, tmp_path):
    run_orig = RunDirectory(mock_fxe_raw_run)
    voview_file = tmp_path / 'overview.h5'

    vofw = voview.VirtualOverviewFileWriter(voview_file, run_orig)
    vofw.write()

    vf = H5File(voview_file)
    cam_src = vf['FXE_XAD_GEC/CAM/CAMERA_NODATA:daqOutput']
    assert cam_src.aggregator == 'DA01'
    cam_data = cam_src['data.image.pixels']
    assert cam_data.shape[0] == 0
    assert cam_data.source_file_paths == [
        Path(mock_fxe_raw_run, 'RAW-R0450-DA01-S00000.h5')
    ]


def open_run_with_voview(run_src, new_run_dir):
    copytree(run_src, new_run_dir)
    voview_file = new_run_dir / 'overview.h5'
    run_orig = RunDirectory(str(new_run_dir), _use_voview=False)
    vofw = voview.VirtualOverviewFileWriter(voview_file, run_orig)
    vofw.write()
    opened = RunDirectory(str(new_run_dir))
    assert len(opened.files) == 1
    return opened


def test_combine_voview(mock_spb_raw_run, mock_spb_proc_run, tmp_path):
    raw_dc = open_run_with_voview(mock_spb_raw_run, tmp_path / 'r0238_raw')
    proc_dc = open_run_with_voview(mock_spb_proc_run, tmp_path / 'r0238_proc')

    # Deselect & union data like we do for open_run(..., data='all')
    raw_extra = raw_dc.deselect([
        (src, '*') for src in raw_dc.all_sources & proc_dc.all_sources]
    )
    assert raw_extra.instrument_sources == {
        'SA1_XTD2_XGM/DOOCS/MAIN:output',
        'SPB_XTD9_XGM/DOOCS/MAIN:output',
        'SPB_IRU_CAM/CAM/SIDEMIC:daqOutput',
    }
    run = proc_dc.union(raw_extra)

    assert 'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf' in run.instrument_sources
    assert 'SA1_XTD2_XGM/DOOCS/MAIN' in run.control_sources


def test_voview_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(voview, 'DATA_ROOT_DIR', str(tmp_path))

    maxwell_run_dir = tmp_path / 'raw' / 'XMPL' / '202102' / 'p700000' / 'r0123'
    maxwell_run_dir.mkdir(parents=True)
    voview_file_in_run_m = maxwell_run_dir / 'overview.h5'
    usr_dir = tmp_path / 'XMPL' / '202102' / 'p700000' / 'usr'
    usr_dir.mkdir(parents=True)
    voview_file_in_usr = usr_dir / '.extra_data' / 'RAW-R0123-OVERVIEW.h5'

    assert voview.voview_paths_for_run(maxwell_run_dir) == [
        str(voview_file_in_run_m), str(voview_file_in_usr)
    ]

    online_run_dir = tmp_path / 'XMPL' / '202102' / 'p700000' / 'raw' / 'r0123'
    online_run_dir.mkdir(parents=True)
    voview_file_in_run_o = online_run_dir / 'overview.h5'

    assert voview.voview_paths_for_run(online_run_dir) == [
        str(voview_file_in_run_o), str(voview_file_in_usr)
    ]
