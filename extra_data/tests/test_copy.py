from pathlib import Path

import h5py
import numpy as np

from extra_data.copy import copy_structure, main


def test_copy_structure(tmp_path, mock_sa3_control_data):
    xgm = "SA3_XTD10_XGM/XGM/DOOCS"
    xgm_intensity = f"INSTRUMENT/{xgm}:output/data/intensityTD"
    xgm_flux = f"CONTROL/{xgm}/pulseEnergy/photonFlux/value"

    ext_file = 'ext-data.h5'
    ext_path = 'some/data'
    with h5py.File(mock_sa3_control_data, "a") as f:
        # add some data
        ds = f[xgm_intensity]
        ds[:] = np.ones(ds.shape, ds.dtype)
        ds = f[xgm_flux]
        ds[:] = np.ones(ds.shape, ds.dtype)
        # add softlink
        f["group/SOFTLINKED"] = h5py.SoftLink(f"/{xgm_intensity}")
        # add hardlink
        f['group/HARDLINKED'] = ds
        # add external link
        with h5py.File(Path(mock_sa3_control_data).parent / ext_file, 'w') as g:
            g[ext_path] = [1]
        f['group/EXTLINK'] = h5py.ExternalLink(ext_file, ext_path)

    copy_structure(mock_sa3_control_data, tmp_path, control_data=True)

    inp = h5py.File(mock_sa3_control_data)
    out = h5py.File(tmp_path / mock_sa3_control_data.rpartition("/")[-1])
    slink = out.get("group/SOFTLINKED", getlink=True)
    extlink = out.get('group/EXTLINK', getlink=True)

    # softlinks are copied
    assert isinstance(slink, h5py.SoftLink)
    assert slink.path == f"/{xgm_intensity}"
    # hardlink
    assert out['group/HARDLINKED'] == out[xgm_flux]
    # external link
    assert extlink.filename == ext_file
    assert extlink.path == ext_path
    # data is not copied
    assert out[xgm_intensity].shape == inp[xgm_intensity].shape
    assert out[xgm_intensity].dtype == inp[xgm_intensity].dtype
    assert (out[xgm_intensity][()] == 0).all()
    # attributes are copied
    assert out[xgm_intensity].attrs["unitName"] == "joule"
    # control data is copied
    assert out[xgm_flux].shape == inp[xgm_flux].shape
    assert out[xgm_flux].dtype == inp[xgm_flux].dtype
    assert (out[xgm_flux][()] == 1).all()
    # run data is not copied
    assert out[f"RUN/{xgm}/classId/value"].dtype == h5py.string_dtype()
    assert out[f"RUN/{xgm}/classId/value"][()] == [b""]


def test_copy_run(tmp_path, mock_spb_proc_run):
    copy_structure(mock_spb_proc_run, tmp_path)

    inp_files = list(Path(mock_spb_proc_run).glob('*.h5'))
    out_files = list(tmp_path.glob('*.h5'))
    assert len(inp_files) == len(out_files)


def test_cli(tmp_path, mock_scs_run):
    # smoke test
    main([mock_scs_run, str(tmp_path)])
