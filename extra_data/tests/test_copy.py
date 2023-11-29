import h5py
import numpy as np

from extra_data.copy import copy_structure


def test_copy_structure(tmp_path, mock_sa3_control_data):
    xgm = "SA3_XTD10_XGM/XGM/DOOCS"
    xgm_intensity = f"INSTRUMENT/{xgm}:output/data/intensityTD"
    xgm_flux = f"CONTROL/{xgm}/pulseEnergy/photonFlux/value"
    with h5py.File(mock_sa3_control_data, "a") as f:
        # add softlink
        f[f"LINKED/{xgm_intensity}"] = h5py.SoftLink(f"/{xgm_intensity}")
        # add some data
        ds = f[xgm_intensity]
        ds[:] = np.ones(ds.shape, ds.dtype)
        ds = f[xgm_flux]
        ds[:] = np.ones(ds.shape, ds.dtype)

    copy_structure(mock_sa3_control_data, tmp_path, control_data=True)

    inp = h5py.File(mock_sa3_control_data)
    out = h5py.File(tmp_path / mock_sa3_control_data.rpartition("/")[-1])
    slink = out.get(f"LINKED/{xgm_intensity}", getlink=True)

    # softlinks are copied
    assert isinstance(slink, h5py.SoftLink)
    assert slink.path == f"/{xgm_intensity}"
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

    # TODO test hardlinks


def test_copy_run():
    ...
