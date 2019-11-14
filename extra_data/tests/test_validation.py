from extra_data import RunDirectory
from extra_data import validation


def test_validate_run(mock_fxe_raw_run):
    rv = validation.RunValidator(mock_fxe_raw_run)
    rv.validate()
