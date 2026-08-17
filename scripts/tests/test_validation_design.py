"""Regression tests for validation-sample allocation and weighted scoring."""
import numpy as np
import pandas as pd
import pytest

import validate


def _corpus(n, found):
    return pd.DataFrame({
        'id': range(n),
        'year': [1910 + 10 * (i % 10) for i in range(n)],
        'raw_content': ['job ad'] * n,
        'addresses': [[{'city': 'Norfolk'}] if i < found else []
                      for i in range(n)],
        'wage': [None] * n,
    })


def test_allocation_hits_exact_target_when_one_side_is_sparse_or_absent():
    for population, requested, expected in [
            (_corpus(101, found=1), 21, 21),
            (_corpus(101, found=101), 21, 21),
            (_corpus(17, found=9), 30, 17)]:
        sample = validate.draw_sample(population, requested, seed=17)
        assert len(sample) == expected
        assert sample.index.is_unique
        assert sample.design_weight.sum() == pytest.approx(len(population))


def test_allocation_rejects_nonpositive_or_unidentified_designs():
    population = _corpus(100, found=50)
    with pytest.raises(ValueError, match='must be positive'):
        validate.draw_sample(population, 0, seed=1)
    with pytest.raises(ValueError, match='must be an integer'):
        validate.draw_sample(population, 20.5, seed=1)
    # Twenty nonempty decade x found cells need at least one draw each.
    with pytest.raises(ValueError, match='smaller than the 20 nonempty strata'):
        validate.draw_sample(population, 19, seed=1)


def test_strata_use_one_advancing_rng_not_identical_per_cell_draws():
    rows = []
    for cell in range(6):
        for position in range(20):
            rows.append({
                'year': 1940 + cell * 10,
                'position': position,
                'raw_content': 'job ad',
                'addresses': [{'city': 'Norfolk'}],
            })
    sample = validate.draw_sample(pd.DataFrame(rows), 12, seed=29)
    repeated = validate.draw_sample(pd.DataFrame(rows), 12, seed=29)
    assert sample.index.to_list() == repeated.index.to_list()
    positions = [tuple(sorted(group.position))
                 for _, group in sample.groupby('_decade')]
    assert len(set(positions)) > 1, (
        'within-cell draws are synchronized because every stratum reset the RNG')


def test_missing_year_and_missing_address_cells_are_handled_explicitly():
    df = pd.DataFrame({
        'year': [None, np.nan, 1975],
        'raw_content': ['a', 'b', 'c'],
        'addresses': [None, np.nan, [{'city': 'Norfolk'}]],
    })
    sample = validate.draw_sample(df, 3, seed=4)
    assert {'no-year|empty', '1970|found'} == set(sample.stratum)
    with pytest.raises(ValueError, match='list/array'):
        validate.draw_sample(df.assign(addresses=['malformed'] * 3), 2, seed=4)


def test_weighted_wilson_contains_exact_boundary_points():
    weights = pd.Series([2.5, 7.0, 3.0])
    denominator = pd.Series([True, True, True])
    all_one = validate.weighted_rate(denominator, denominator, weights)
    all_zero = validate.weighted_rate(~denominator, denominator, weights)
    assert all_one[0] == 1.0 and all_one[2] == 1.0
    assert all_zero[0] == 0.0 and all_zero[1] == 0.0


def _blank_template():
    data = {
        'stratum': ['1970|found', '1970|found'],
        'stratum_size': [10, 10],
        'stratum_drawn': [2, 2],
        'design_weight': [5.0, 5.0],
    }
    for column in validate.CODING_COLUMNS:
        data[column] = ['', '']
    return pd.DataFrame(data)


def test_template_rejects_edited_weights_and_deleted_rows():
    template = _blank_template()
    validate.validate_template(template)

    edited = template.copy()
    edited.loc[0, 'design_weight'] = 4.0
    with pytest.raises(SystemExit, match='must equal'):
        validate.validate_template(edited)

    with pytest.raises(SystemExit, match='deleted or duplicated'):
        validate.validate_template(template.iloc[:1])
