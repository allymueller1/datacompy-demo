#
# Copyright 2026 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from decimal import Decimal

import pandas as pd
import polars as pl
from datacompy.comparator.numeric import (
    PandasNumericComparator,
    PolarsNumericComparator,
    decimal_comparator,
)


def test_decimal_comparator_matches_any_decimal_precision():
    d = decimal_comparator()
    assert isinstance(d, str) and d == "decimal"
    assert d == "decimal(10,2)"
    assert d == "decimal(38,0)"
    assert d == "decimal"
    assert "decimal(5,1)" == d  # noqa: SIM300
    assert "decimal(12,4)" in [d]


def test_decimal_comparator_rejects_other_types():
    d = decimal_comparator()
    assert d != "double"
    assert d != "int"
    assert d != "dec"
    assert "bigint" not in [d]


# tests for PolarsNumericComparator
def test_polars_numeric_comparator_exact_match():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.0, 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_approximate_match():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.001, 2.002, 3.003])
    result = comparator.compare(col1, col2, rtol=1e-3, atol=1e-3)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_type_casting():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1, 2, 3])  # Integer type
    col2 = pl.Series([1.0, 2.0, 3.0])  # Float type
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_nan_handling():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, float("nan"), 3.0])
    col2 = pl.Series([1.0, float("nan"), 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_numeric_comparator_mismatch():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.5, 3.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, False, True]


def test_polars_numeric_comparator_error_handling():
    comparator = PolarsNumericComparator()
    col1 = pl.Series(["a", "b", "c"])  # Invalid type for numeric comparison
    col2 = pl.Series(["x", "y", "z"])  # Invalid type for numeric comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pl.Series([1, 2, 3])
    col2 = pl.Series([1, 2, 3, 4])
    result = comparator.compare(col1, col2)
    assert result is None


def test_polars_numeric_comparator_invalid_tolerance_returns_all_false():
    """A tolerance NumPy cannot use returns all False instead of raising."""
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 2.0, 3.0])
    col2 = pl.Series([1.0, 2.0, 3.0])
    assert comparator.compare(col1, col2, rtol=None).to_list() == [
        False,
        False,
        False,
    ]
    assert comparator.compare(col1, col2, atol="x").to_list() == [
        False,
        False,
        False,
    ]


def test_polars_numeric_comparator_empty_series():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([], dtype=pl.Float64)
    col2 = pl.Series([], dtype=pl.Float64)
    result = comparator.compare(col1, col2)
    assert result.to_list() == []
    assert result.dtype == pl.Boolean


def test_polars_numeric_comparator_tolerance_boundary_inclusive():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 1.0])
    col2 = pl.Series([1.5, 1.5000001])
    assert comparator.compare(col1, col2, rtol=0, atol=0.5).to_list() == [
        True,
        False,
    ]


def test_polars_numeric_comparator_relative_tolerance_uses_second_column():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([10.0, 12.0])
    col2 = pl.Series([12.0, 10.0])
    assert comparator.compare(col1, col2, rtol=0.18, atol=0).to_list() == [
        True,
        False,
    ]


def test_polars_numeric_comparator_nan_vs_number():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([float("nan"), 1.0])
    col2 = pl.Series([1.0, float("nan")])
    assert comparator.compare(col1, col2).to_list() == [False, False]


def test_polars_numeric_comparator_null_handling():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, None, None])
    col2 = pl.Series([1.0, None, 2.0])
    assert comparator.compare(col1, col2).to_list() == [True, True, False]


def test_polars_numeric_comparator_decimal():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([Decimal("1.10"), Decimal("2.20")])
    col2 = pl.Series([Decimal("1.10"), Decimal("2.25")])
    assert comparator.compare(col1, col2).to_list() == [True, False]


# tests for PandasNumericComparator
def test_pandas_numeric_comparator_exact_match():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.0, 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_approximate_match():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.001, 2.002, 3.003])
    result = comparator.compare(col1, col2, rtol=1e-3, atol=1e-3)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_type_casting():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1, 2, 3])  # Integer type
    col2 = pd.Series([1.0, 2.0, 3.0])  # Float type
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_nan_handling():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, float("nan"), 3.0])
    col2 = pd.Series([1.0, float("nan"), 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_numeric_comparator_mismatch():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.5, 3.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, False, True]


def test_pandas_numeric_comparator_error_handling():
    comparator = PandasNumericComparator()
    col1 = pd.Series(["a", "b", "c"])  # Invalid type for numeric comparison
    col2 = pd.Series(["x", "y", "z"])  # Invalid type for numeric comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.5, 3.0, 4.0])
    result = comparator.compare(col1, col2)
    assert result is None


def test_pandas_numeric_comparator_invalid_tolerance_returns_all_false():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 2.0, 3.0])
    col2 = pd.Series([1.0, 2.0, 3.0])
    assert comparator.compare(col1, col2, rtol=None).tolist() == [
        False,
        False,
        False,
    ]
    assert comparator.compare(col1, col2, atol="x").tolist() == [
        False,
        False,
        False,
    ]


def test_pandas_numeric_comparator_uncastable_integers_return_all_false():
    """An uncastable integer marks every row mismatched and preserves the index."""
    comparator = PandasNumericComparator()
    col1 = pd.Series([10**400, 1], dtype=object, index=[5, 6])
    col2 = pd.Series([1, 1], dtype=object, index=[5, 6])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [False, False]
    assert result.index.tolist() == [5, 6]


def test_pandas_numeric_comparator_empty_series():
    comparator = PandasNumericComparator()
    col1 = pd.Series([], dtype=float)
    col2 = pd.Series([], dtype=float)
    result = comparator.compare(col1, col2)
    assert result.tolist() == []
    assert result.dtype == bool


def test_pandas_numeric_comparator_tolerance_boundary_inclusive():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 1.0])
    col2 = pd.Series([1.5, 1.5000001])
    assert comparator.compare(col1, col2, rtol=0, atol=0.5).tolist() == [
        True,
        False,
    ]


def test_pandas_numeric_comparator_relative_tolerance_uses_second_column():
    comparator = PandasNumericComparator()
    col1 = pd.Series([10.0, 12.0])
    col2 = pd.Series([12.0, 10.0])
    assert comparator.compare(col1, col2, rtol=0.18, atol=0).tolist() == [
        True,
        False,
    ]


def test_pandas_numeric_comparator_nan_vs_number():
    comparator = PandasNumericComparator()
    col1 = pd.Series([float("nan"), 1.0])
    col2 = pd.Series([1.0, float("nan")])
    assert comparator.compare(col1, col2).tolist() == [False, False]


def test_pandas_numeric_comparator_nullable_integers():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1, None, None, 3], dtype="Int64")
    col2 = pd.Series([1, None, 2, 4], dtype="Int64")
    assert comparator.compare(col1, col2).tolist() == [True, True, False, False]


def test_pandas_numeric_comparator_decimal_vs_float():
    comparator = PandasNumericComparator()
    col1 = pd.Series([Decimal("1.10"), Decimal("2.20")])
    col2 = pd.Series([1.1, 2.25])
    assert comparator.compare(col1, col2).tolist() == [True, False]
