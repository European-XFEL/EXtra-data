
import pytest

from tempfile import TemporaryDirectory
from . import make_examples


@pytest.mark.parametrize(
    ['rep_rate', 'gain_setting', 'integration_time', 'bias_voltage'],
    [(False, False, False, False), (True, True, True, True)],
    ids=['without-keys', 'with-keys'])
def test_agipd1m_runs(rep_rate, gain_setting, integration_time, bias_voltage):
    with TemporaryDirectory() as td:
        make_examples.make_agipd1m_run(
            td, rep_rate=rep_rate, gain_setting=gain_setting,
            integration_time=integration_time, bias_voltage=bias_voltage)


def test_agipd500k_runs():
    with TemporaryDirectory() as td:
        make_examples.make_agipd500k_run(td)
