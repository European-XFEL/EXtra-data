
from extra_data.cli.check_readable import main


def test_check_readable(mock_spb_proc_run, capsys):
    assert main([mock_spb_proc_run]) == 0

    captured = capsys.readouterr()
    out_lines = captured.out.splitlines()

    assert len(out_lines) == 3
    assert set(out_lines[0]) == {'.'}
    assert out_lines[-1] == 'all files readable'


def test_check_readable_all(mock_spb_proc_run, capsys):
    assert main([mock_spb_proc_run, '-a']) == 0

    captured = capsys.readouterr()
    out_lines = captured.out.splitlines()

    assert len(out_lines) == 18
    assert set(out_lines[0]) == {'.'}
