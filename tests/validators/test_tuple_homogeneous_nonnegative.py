import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pydantic_core import SchemaValidator, ValidationError, core_schema


class IntSubclass(int):
    def __index__(self):
        raise AssertionError('int subclasses retain native integer semantics')


def validator(strict, referenced):
    item = core_schema.int_schema(ge=0, strict=strict, ref='nonnegative' if referenced else None)
    schema = core_schema.tuple_schema(
        [core_schema.definition_reference_schema('nonnegative') if referenced else item],
        variadic_item_index=0,
    )
    return SchemaValidator(core_schema.definitions_schema(schema, [item]) if referenced else schema)


def outcome(v, values, mode):
    try:
        return ('ok', getattr(v, 'validate_' + mode)(values))
    except ValidationError as exc:
        return ('error', exc.errors(include_url=False))


@pytest.mark.parametrize('strict', [True, False])
@given(st.lists(st.integers(min_value=-(2**200), max_value=2**200), max_size=40))
def test_exact_nonnegative_reference_matches_general(strict, values):
    direct = validator(strict, False)
    referenced = validator(strict, True)
    assert outcome(referenced, tuple(values), 'python') == outcome(direct, tuple(values), 'python')
    payload = json.dumps(values)
    assert outcome(referenced, payload, 'json') == outcome(direct, payload, 'json')


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize(
    'values',
    [
        (0, 1, 100278, 2**63 - 1, 2**63, 2**200),
        (-(2**200), -(2**63) - 1, -(2**63), -1, 0, 2**200),
        (True, False, '4', 5.0, 5.5, None, b'6'),
        (IntSubclass(7), IntSubclass(-2), 0),
    ],
)
def test_reference_boundaries_and_error_order(strict, values):
    assert outcome(validator(strict, True), values, 'python') == outcome(validator(strict, False), values, 'python')


def test_exact_integer_identity_and_fresh_tuple():
    values = (1234567, 2**63 - 1, 2**200)
    v = validator(True, True)
    for _ in range(3):
        result = v.validate_python(values)
        assert result == values
        assert result is not values
        assert all(a is b for a, b in zip(values, result))
        assert v.validate_python((0,)) == (0,)
        with pytest.raises(ValidationError):
            v.validate_python((-1,))


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('override', [None, True, False])
@pytest.mark.parametrize('minimum', [None, 0, 1, 4])
@pytest.mark.parametrize('maximum,fail_fast', [(None, False), (None, True), (1, False)])
def test_streamed_tuple_limits_and_strict_override(strict, override, minimum, maximum, fail_fast):
    def build(referenced):
        item = core_schema.int_schema(ge=0, strict=strict, ref='n' if referenced else None)
        schema = core_schema.tuple_schema(
            [core_schema.definition_reference_schema('n') if referenced else item],
            variadic_item_index=0,
            min_length=minimum,
            max_length=maximum,
            fail_fast=fail_fast,
        )
        if referenced:
            schema = core_schema.definitions_schema(schema, [item])
        return SchemaValidator(schema)

    def result(v, values):
        try:
            return ('ok', v.validate_python(values, strict=override))
        except ValidationError as exc:
            return ('error', exc.errors(include_url=False))

    direct, referenced = build(False), build(True)
    for values in [(), (0,), (1, 2, 3), (-1, -2, 3), (None, True, '3'), (2**200, IntSubclass(7))]:
        assert result(referenced, values) == result(direct, values)
        assert result(referenced, list(values)) == result(direct, list(values))


def test_tuple_subclass_keeps_native_iteration_behavior():
    class CustomTuple(tuple):
        def __iter__(self):
            raise AssertionError('Python tuple slots, not overridden iteration')

    for strict in [True, False]:
        values = CustomTuple((1, 2, 3))
        assert validator(strict, True).validate_python(values) == validator(strict, False).validate_python(values)
