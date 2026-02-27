import pytest

from ..display import SourceGroup

@pytest.mark.parametrize(("name", "exp"), [
    ("SPB_IRU_INLINEMIC/MOTOR/MIC_AX", [
        "", "SPB", "_", "IRU", "_", "INLINEMIC", "/MOTOR/", "MIC", "_", "AX", ""
    ]),
    ("SA1_XTD9_HIREX/DET/GOTTHARD2_MASTER:daqOutput", [
        "", "SA", "", "1", "_", "XTD", "", "9", "_", "HIREX", "/DET/",
        "GOTTHARD", "", "2", "_", "MASTER", ":", "daqOutput", "",
    ]),
    ("XFEL_SYNC/LINK_LOCK/XHEXP1_AMC5_ACTUATOR", [
        "", "XFEL", "_", "SYNC", "/", "LINK", "_", "LOCK", "/",
        "XHEXP", "", "1", "_", "AMC", "", "5", "_", "ACTUATOR", ""
    ]),
    ("SQS_RACK_MPOD-1/MDL/MPOD_MAPPER", [
        "", "SQS", "_", "RACK", "_", "MPOD", "-", "1", "/MDL/", "MPOD", "_", "MAPPER", ""
    ]),
    ("LAS_PPL_SA3XT/MOTOR/SMARACTCMCOMPSTAGE", [
        "", "LAS", "_", "PPL", "_", "SA", "", "3", "", "XT", "/MOTOR/SMARACTCMCOMPSTAGE"
    ]),
    ("FXE_OGT2_BIU-2/CAM/CAMERA", [
        "", "FXE", "_", "OGT", "", "2", "_", "BIU", "-", "2", "/CAM/CAMERA",
    ])
])
def test_split_name(name, exp):
    assert SourceGroup.split_name(name) == exp


@pytest.mark.parametrize(("exp", "names"), [
    ("SPB_IRU_INLINEMIC/MOTOR/MIC_{AX, AY, X, Y, Z}", [
        "SPB_IRU_INLINEMIC/MOTOR/MIC_AX",
        "SPB_IRU_INLINEMIC/MOTOR/MIC_AY",
        "SPB_IRU_INLINEMIC/MOTOR/MIC_X",
        "SPB_IRU_INLINEMIC/MOTOR/MIC_Y",
        "SPB_IRU_INLINEMIC/MOTOR/MIC_Z",
    ]),
    ("SA1_XTD9_HIREX/DET/GOTTHARD2_{FOLLOWER, LEADER}:daqOutput", [
        "SA1_XTD9_HIREX/DET/GOTTHARD2_LEADER:daqOutput",
        "SA1_XTD9_HIREX/DET/GOTTHARD2_FOLLOWER:daqOutput",
    ]),
    ("XFEL_SYNC/LINK_LOCK/XHEXP1_AMC5_{ACTUATOR, CONTROLLER}", [
        "XFEL_SYNC/LINK_LOCK/XHEXP1_AMC5_ACTUATOR",
        "XFEL_SYNC/LINK_LOCK/XHEXP1_AMC5_CONTROLLER",
    ]),
    ("SQS_RACK_MPOD-{1-2}/MDL/MPOD_MAPPER", [
        "SQS_RACK_MPOD-1/MDL/MPOD_MAPPER",
        "SQS_RACK_MPOD-2/MDL/MPOD_MAPPER",
    ]),
    ("FXE_AUXT_LIC/DOOCS/BAM_1932{M, S}:output", [
        "FXE_AUXT_LIC/DOOCS/BAM_1932M:output",
        "FXE_AUXT_LIC/DOOCS/BAM_1932S:output",
    ]),
    ("FXE_DET_MOV/MOTOR/{X, Y, Z}", [
        "FXE_DET_MOV/MOTOR/X",
        "FXE_DET_MOV/MOTOR/Y",
        "FXE_DET_MOV/MOTOR/Z",
    ]),
    ("FXE_SMS_USR/MOTOR/UM{01-03, 08}", [
        "FXE_SMS_USR/MOTOR/UM01",
        "FXE_SMS_USR/MOTOR/UM02",
        "FXE_SMS_USR/MOTOR/UM03",
        "FXE_SMS_USR/MOTOR/UM08",
    ]),
])
def test_group_ok(names, exp):
    grp = SourceGroup()
    for name in names:
        assert grp.add(name) is True
    assert str(grp) == exp


@pytest.mark.parametrize("names", [
    [
        # Whole name component changing
        "LAS_PPL_SA3XT/MOTOR/SMARACTCMCOMPSTAGE",
        "LAS_PPL_SA3XT/MOTOR/SMARACTDMCOMPSTAGE",
    ],
    [
        "FXE_OGT2_BIU-2/CAM/CAMERA",
        "FXE_OGT2_BIU-2/IMGPROC/CAMERA",
    ],
    [
        # Multiple parts change
        "SA1_XTD2_XGM/DOOCS/MAIN:output",
        "SPB_XTD9_XGM/DOOCS/MAIN:output",
    ],
])
def test_group_fail(names):
    grp = SourceGroup()
    for name in names[:-1]:
        assert grp.add(name) is True
    assert grp.add(names[-1]) is False
