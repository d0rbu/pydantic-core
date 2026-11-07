"""The shared-definition tuple route used by immutable clean-token rows."""

import gc
import re

import pytest

from pydantic_core import PydanticOmit, SchemaValidator, ValidationError, core_schema


def shared_tuple(item=None, **kwargs):
    if item is None:
        item = core_schema.int_schema(ge=0)
    definition = {**item, 'ref': 'item'}
    reference = core_schema.definition_reference_schema('item')
    validator = SchemaValidator(
        core_schema.definitions_schema(
            core_schema.typed_dict_schema(
                {
                    'tokens': core_schema.typed_dict_field(
                        core_schema.tuple_schema([reference], variadic_item_index=0, **kwargs)
                    ),
                    # A second actual consumer prevents single-use reference inlining.
                    'witness': core_schema.typed_dict_field(reference, required=False),
                }
            ),
            [definition],
        )
    )
    assert re.search(r'TupleValidator\s*\{.*?validators:\s*\[\s*DefinitionRef\(', repr(validator), re.S)
    return validator


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize('container', [tuple, list, iter, TupleSubclass])
def test_shared_tuple_values_and_fresh_output(container):
    values = (10000, 10000, 2**80)
    supplied = container(values)
    result = shared_tuple(min_length=1).validate_python({'tokens': supplied})['tokens']
    assert result == values
    assert type(result) is tuple
    assert result is not supplied
    assert all(output is original for output, original in zip(result, values))


@pytest.mark.parametrize('strict', [True, False])
def test_shared_tuple_json_and_python(strict):
    validator = shared_tuple(strict=strict, min_length=1)
    assert validator.validate_python({'tokens': (0, 7, 2**80)})['tokens'] == (0, 7, 2**80)
    assert validator.validate_json(b'{"tokens":[0,7,1208925819614629174706176]}')['tokens'] == (0, 7, 2**80)
    if strict:
        with pytest.raises(ValidationError) as exc:
            validator.validate_python({'tokens': [1, 2]})
        assert exc.value.errors(include_url=False)[0]['type'] == 'tuple_type'
    else:
        assert validator.validate_python({'tokens': ['1', b'2']})['tokens'] == (1, 2)


@pytest.mark.parametrize('fail_fast', [True, False])
def test_shared_tuple_error_locations_and_iteration_stop(fail_fast):
    consumed = []

    def values():
        for value in (3, -2, 'bad', 7):
            consumed.append(value)
            yield value

    with pytest.raises(ValidationError) as exc:
        shared_tuple(fail_fast=fail_fast).validate_python({'tokens': values()})
    errors = exc.value.errors(include_url=False)
    assert [(error['type'], error['loc']) for error in errors] == (
        [('greater_than_equal', ('tokens', 1))]
        if fail_fast
        else [('greater_than_equal', ('tokens', 1)), ('int_parsing', ('tokens', 2))]
    )
    assert consumed == ([3, -2] if fail_fast else [3, -2, 'bad', 7])


def test_shared_tuple_callbacks_omit_defaults_and_order():
    seen = []

    def convert(value):
        seen.append(value)
        if value == 'omit':
            raise PydanticOmit
        return value

    item = core_schema.no_info_before_validator_function(
        convert, core_schema.with_default_schema(core_schema.int_schema(ge=0), default=11, on_error='default')
    )
    result = shared_tuple(item).validate_python({'tokens': ('3', 'omit', -2, '5'), 'witness': '9'})
    assert result == {'tokens': (3, 11, 5), 'witness': 9}
    assert seen == ['3', 'omit', -2, '5', '9']


@pytest.mark.parametrize('mode', ['python', 'json'])
def test_shared_tuple_partial_validation_still_reports_last_error(mode):
    validator = shared_tuple()
    with pytest.raises(ValidationError) as exc:
        if mode == 'python':
            validator.validate_python({'tokens': (1, -1)}, allow_partial=True)
        else:
            validator.validate_json(b'{"tokens":[1,-1]}', allow_partial=True)
    assert exc.value.errors(include_url=False)[0]['loc'] == ('tokens', 1)


def test_shared_tuple_length_checks_remain_after_validation():
    validator = shared_tuple(min_length=2, max_length=3)
    with pytest.raises(ValidationError) as exc:
        validator.validate_python({'tokens': ()})
    assert exc.value.errors(include_url=False)[0]['type'] == 'too_short'
    with pytest.raises(ValidationError) as exc:
        validator.validate_python({'tokens': (1, 2, 3, 4)})
    assert exc.value.errors(include_url=False)[0]['type'] == 'too_long'


def test_shared_tuple_callback_reentry_and_item_guard_release():
    seen = []
    validator = None

    def check(value):
        seen.append(value)
        if value == 1:
            assert validator.validate_python({'tokens': (101, 101)})['tokens'] == (101, 101)
        return value

    validator = shared_tuple(core_schema.no_info_before_validator_function(check, core_schema.int_schema()))
    assert validator.validate_python({'tokens': (1, 2, 2)})['tokens'] == (1, 2, 2)
    assert seen == [1, 101, 101, 2, 2]


def test_shared_tuple_callback_can_release_external_owner():
    owner = {}
    seen = []

    def check(value):
        owner.clear()
        gc.collect()
        seen.append(value)
        return value

    owner['validator'] = shared_tuple(core_schema.no_info_before_validator_function(check, core_schema.int_schema()))
    assert owner['validator'].validate_python({'tokens': (1, 2)}) == {'tokens': (1, 2)}
    assert owner == {}
    assert seen == [1, 2]


def test_shared_tuple_iteration_failure_is_not_swallowed():
    def values():
        yield 1
        raise RuntimeError('iteration stopped')

    with pytest.raises(ValidationError) as exc:
        shared_tuple().validate_python({'tokens': values()})
    error = exc.value.errors(include_url=False)[0]
    assert error['type'] == 'iteration_error'
    assert 'iteration stopped' in error['msg']


@pytest.mark.parametrize(
    'bounds,expected_calls',
    [({}, [1, 2, 3]), ({'max_length': 1}, [1, 2]), ({'min_length': 4}, [1, 2, 3])],
)
def test_shared_tuple_callback_outputs_release_on_error(bounds, expected_calls):
    calls = []
    released = []

    class Tracked:
        def __init__(self, value):
            self.value = value

        def __del__(self):
            released.append(self.value)

    def produce(value):
        calls.append(value)
        if value == 3:
            raise ValueError('third item')
        return Tracked(value)

    validator = shared_tuple(core_schema.no_info_plain_validator_function(produce), **bounds)
    with pytest.raises(ValidationError):
        validator.validate_python({'tokens': (1, 2, 3)})
    assert calls == expected_calls
    assert released == [1, 2]


def test_shared_recursive_tuple_preserves_cycle_and_depth_guards():
    reference = core_schema.definition_reference_schema('node')
    validator = shared_tuple(
        core_schema.definitions_schema(
            reference,
            [
                core_schema.union_schema(
                    [core_schema.int_schema(), core_schema.tuple_schema([reference], variadic_item_index=0)],
                    ref='node',
                )
            ],
        )
    )
    assert validator.validate_python({'tokens': (1, (2, (3,)))})['tokens'] == (1, (2, (3,)))
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValidationError) as exc:
        validator.validate_python({'tokens': (cycle,)})
    assert 'recursion_loop' in {error['type'] for error in exc.value.errors(include_url=False)}
    nested = 1
    for _ in range(300):
        nested = (nested,)
    with pytest.raises(ValidationError) as exc:
        validator.validate_python({'tokens': (nested,)})
    assert 'recursion_loop' in {error['type'] for error in exc.value.errors(include_url=False)}
    # Failed recursion cannot poison later calls or repeated identities.
    assert validator.validate_python({'tokens': (1, 1)})['tokens'] == (1, 1)


@pytest.mark.parametrize('with_head', [True, False])
def test_shared_tuple_head_and_tail_keep_the_original_path(with_head):
    reference = core_schema.definition_reference_schema('item')
    items = [core_schema.str_schema(), reference, reference] if with_head else [reference, core_schema.str_schema()]
    validator = SchemaValidator(
        core_schema.definitions_schema(
            core_schema.tuple_schema(items, variadic_item_index=1 if with_head else 0),
            [core_schema.int_schema(ref='item')],
        )
    )
    if with_head:
        assert validator.validate_python(('start', '1', '2', '3')) == ('start', 1, 2, 3)
    else:
        assert validator.validate_python(('1', '2', 'end')) == (1, 2, 'end')


@pytest.mark.parametrize('strict', [False, True])
def test_nonnegative_shared_reference_preserves_exact_and_large_integer_identity(strict):
    values = (0, 10000, 10000, 2**63 - 1, 2**63, 10**700)
    validator = shared_tuple(core_schema.int_schema(ge=0, strict=strict))
    output = validator.validate_python({'tokens': values})['tokens']
    assert output is not values
    assert all(a is b for a, b in zip(output, values))
    assert validator.validate_python({'tokens': (10000,) * 4096})['tokens'] == (10000,) * 4096
    with pytest.raises(ValidationError) as exc:
        validator.validate_python({'tokens': (1, -(2**80), -1)})
    assert [e['loc'] for e in exc.value.errors(include_url=False)] == [('tokens', 1), ('tokens', 2)]
    assert validator.validate_python({'tokens': (10000, 10000)})['tokens'] == (10000, 10000)


@pytest.mark.parametrize('strict', [False, True])
def test_nonnegative_shared_reference_preserves_bool_and_subclass_semantics(strict):
    calls = []

    class Integer(int):
        def __int__(self):
            calls.append('int')
            raise RuntimeError('must not call')

        def __index__(self):
            calls.append('index')
            raise RuntimeError('must not call')

    validator = shared_tuple(core_schema.int_schema(ge=0, strict=strict))
    output = validator.validate_python({'tokens': (Integer(9),)})['tokens']
    assert type(output[0]) is int
    assert output == (9,)
    assert calls == []
    if strict:
        with pytest.raises(ValidationError) as exc:
            validator.validate_python({'tokens': (True,)})
        assert exc.value.errors(include_url=False)[0]['type'] == 'int_type'
    else:
        assert validator.validate_python({'tokens': (True, False)})['tokens'] == (1, 0)


@pytest.mark.parametrize('constraints', [{'ge': 0, 'le': 3}, {'ge': 0, 'lt': 3}, {'ge': 0, 'multiple_of': 2}, {'gt': 0}])
def test_shared_reference_other_constraints_still_reject(constraints):
    invalid = 0 if constraints == {'gt': 0} else 7
    with pytest.raises(ValidationError):
        shared_tuple(core_schema.int_schema(**constraints)).validate_python({'tokens': (invalid,)})
