import numpy as np
import pytest

from extra_data.utils import QuickView


def test_init_quick_view():
    qv = QuickView()

    assert qv.data is None
    qv.data = np.empty((1, 1, 1), dtype=np.int8)
    assert len(qv) == 1
    assert qv.pos == 0

    with pytest.raises(TypeError) as info:
        qv.data = 4

    with pytest.raises(TypeError) as info:
        qv.data = np.empty((1, 1, 1, 1), dtype=np.int8)


if __name__ == "__main__":
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
