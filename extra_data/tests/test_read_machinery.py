import os
import os.path as osp
from unittest import mock

import numpy as np
from extra_data import RunDirectory, read_machinery


def test_find_proposal(tmpdir):
    prop_dir = osp.join(str(tmpdir), 'SPB', '201701', 'p002012')
    os.makedirs(prop_dir)

    with mock.patch.object(read_machinery, 'DATA_ROOT_DIR', str(tmpdir)):
        assert read_machinery.find_proposal('p002012') == prop_dir

        assert read_machinery.find_proposal(prop_dir) == prop_dir


def test_same_run(mock_spb_raw_run, mock_jungfrau_run, mock_scs_run):
    run_spb = RunDirectory(mock_spb_raw_run)
    run_jf = RunDirectory(mock_jungfrau_run)
    run_scs = RunDirectory(mock_scs_run)

    assert run_spb.is_single_run
    assert run_jf.is_single_run
    assert run_scs.is_single_run

    assert not read_machinery.same_run(run_spb, run_scs, run_jf)

    format_version = run_spb.files[0].format_version
    s1 = run_spb.select_trains(np.s_[:1])
    s2 = run_spb.select_trains(np.s_[:-1])
    s3 = run_spb.select_trains(np.s_[3])

    s4 = run_spb.select('SA1_XTD2_XGM/DOOCS/MAIN', '*')
    s5 = run_spb.select('SPB_IRU_CAM/CAM/SIDEMIC:daqOutput', '*')

    if run_spb.run_metadata()['dataFormatVersion'] != '0.5':
        assert read_machinery.same_run(s1, s2, s3)
        assert read_machinery.same_run(s4, s5)
    else:
        assert not read_machinery.same_run(s1, s2, s3)
        assert not read_machinery.same_run(s4, s5)

    # SourceData
    sd = run_spb['SA1_XTD2_XGM/DOOCS/MAIN']
    sd1 = sd.select_trains(np.s_[:1])
    sd2 = sd.select_trains(np.s_[-1:])
    assert sd.is_single_run
    if sd.run_metadata()['dataFormatVersion'] != '0.5':
        assert read_machinery.same_run(sd1, sd2)
    else:
        assert not read_machinery.same_run(sd1, sd2)
