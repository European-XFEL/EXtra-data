import re

from extra_data.cli import lsxfel


def test_lsxfel_file(mock_lpd_data, capsys):
    lsxfel.summarise_file(mock_lpd_data)
    out, err = capsys.readouterr()
    assert "480 trains, 1 source" in out


def test_lsxfel_run(mock_fxe_raw_run, capsys):
    lsxfel.summarise_run(mock_fxe_raw_run)
    out, err = capsys.readouterr()

    assert "480 trains" in out
    assert "16 detector files" in out


def test_lsxfel_main(mock_fxe_raw_run, capsys):
    lsxfel.main([mock_fxe_raw_run, "--source", "XGM"])
    out, err = capsys.readouterr()
    assert re.search(r"trains:\s*480", out)
    assert "SA1_XTD2_XGM/DOOCS/MAIN:output" in out
    assert "FXE_XAD_GEC/CAM/CAMERA" not in out  # Selected out by --source
