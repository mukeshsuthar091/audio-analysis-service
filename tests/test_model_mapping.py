"""Checkpoint output-order regression tests."""

import pytest

from app.inference.model import map_model_output


def test_maps_female_male_child_tensor_order() -> None:
    result = map_model_output(0.37, [0.70, 0.20, 0.10])

    assert result.normalized_age == 0.37
    assert result.female_probability == 0.70
    assert result.male_probability == 0.20
    assert result.child_probability == 0.10


@pytest.mark.parametrize(
    "gender",
    [[], [0.5, 0.5], [0.2, 0.3, 0.5, 0.0], [0.2, float("nan"), 0.8]],
)
def test_rejects_invalid_gender_tensor(gender: list[float]) -> None:
    with pytest.raises(RuntimeError, match="invalid values"):
        map_model_output(0.37, gender)

