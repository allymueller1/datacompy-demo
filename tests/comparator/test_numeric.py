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

import pandas as pd
import polars as pl
from datacompy.comparator.numeric import (
    PandasNumericComparator,
    PolarsNumericComparator,
)


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


def test_decimal_comparator_matches_parameterised_decimal():
    from datacompy.comparator.numeric import decimal_comparator

    comparator = decimal_comparator()
    assert isinstance(comparator, str)
    assert comparator == "decimal"
    assert comparator == "decimal(10,2)"
    assert comparator == "decimal(38,0)"
    assert comparator != "dec"
    assert comparator != "double"


def test_polars_numeric_comparator_empty_series():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([], dtype=pl.Float64)
    col2 = pl.Series([], dtype=pl.Float64)
    result = comparator.compare(col1, col2)
    assert result is not None
    assert result.to_list() == []


def test_polars_numeric_comparator_null_handling():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, None, None])
    col2 = pl.Series([1.0, None, 2.0])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, False]


def test_polars_numeric_comparator_infinity():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([float("inf"), float("-inf"), float("inf")])
    col2 = pl.Series([float("inf"), float("-inf"), float("-inf")])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, False]


def test_polars_numeric_comparator_tolerance_boundaries():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([100.0])
    col2 = pl.Series([101.0])
    assert comparator.compare(col1, col2, rtol=0.01, atol=0).to_list() == [True]
    assert comparator.compare(col1, col2, rtol=0, atol=1).to_list() == [True]
    assert comparator.compare(col1, col2, rtol=0, atol=0.99).to_list() == [False]
    assert comparator.compare(col1, col2, rtol=0, atol=0).to_list() == [False]


def test_polars_numeric_comparator_default_tolerance():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1.0, 1.0])
    col2 = pl.Series([1.000001, 1.0001])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, False]


def test_polars_numeric_comparator_mixed_integer_widths():
    comparator = PolarsNumericComparator()
    col1 = pl.Series([1, 2, 3], dtype=pl.UInt8)
    col2 = pl.Series([1, 2, 4], dtype=pl.Int64)
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, False]


def test_polars_numeric_comparator_decimal_vs_float():
    from decimal import Decimal

    comparator = PolarsNumericComparator()
    col1 = pl.Series([Decimal("1.10"), Decimal("2.20")])
    col2 = pl.Series([1.1, 2.3])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, False]


def test_polars_numeric_comparator_one_side_not_numeric():
    comparator = PolarsNumericComparator()
    assert comparator.compare(pl.Series([1.0, 2.0]), pl.Series(["a", "b"])) is None
    assert comparator.compare(pl.Series(["a", "b"]), pl.Series([1.0, 2.0])) is None
    assert (
        comparator.compare(pl.Series([True, False]), pl.Series([True, False])) is None
    )


def test_polars_numeric_comparator_cast_failure_returns_all_false(monkeypatch):
    import datacompy.comparator.numeric as numeric

    class FailingNumpy:
        @staticmethod
        def isclose(*args, **kwargs):
            raise TypeError("boom")

    monkeypatch.setattr(numeric, "np", FailingNumpy)
    comparator = PolarsNumericComparator()
    result = comparator.compare(pl.Series([1.0, 2.0]), pl.Series([1.0, 2.0]))
    assert result.to_list() == [False, False]


def test_pandas_numeric_comparator_empty_series():
    comparator = PandasNumericComparator()
    col1 = pd.Series([], dtype=float)
    col2 = pd.Series([], dtype=float)
    result = comparator.compare(col1, col2)
    assert result is not None
    assert result.tolist() == []


def test_pandas_numeric_comparator_null_handling():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, None, None])
    col2 = pd.Series([1.0, None, 2.0])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, False]


def test_pandas_numeric_comparator_nullable_integer():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1, 2, 3], dtype="Int64")
    col2 = pd.Series([1, pd.NA, 4], dtype="Int64")
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, False, False]


def test_pandas_numeric_comparator_infinity():
    comparator = PandasNumericComparator()
    col1 = pd.Series([float("inf"), float("-inf"), float("inf")])
    col2 = pd.Series([float("inf"), float("-inf"), float("-inf")])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, False]


def test_pandas_numeric_comparator_tolerance_boundaries():
    comparator = PandasNumericComparator()
    col1 = pd.Series([100.0])
    col2 = pd.Series([101.0])
    assert comparator.compare(col1, col2, rtol=0.01, atol=0).tolist() == [True]
    assert comparator.compare(col1, col2, rtol=0, atol=1).tolist() == [True]
    assert comparator.compare(col1, col2, rtol=0, atol=0.99).tolist() == [False]
    assert comparator.compare(col1, col2, rtol=0, atol=0).tolist() == [False]


def test_pandas_numeric_comparator_default_tolerance():
    comparator = PandasNumericComparator()
    col1 = pd.Series([1.0, 1.0])
    col2 = pd.Series([1.000001, 1.0001])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, False]


def test_pandas_numeric_comparator_decimal_vs_float():
    from decimal import Decimal

    comparator = PandasNumericComparator()
    col1 = pd.Series([Decimal("1.10"), Decimal("2.20")])
    col2 = pd.Series([1.1, 2.3])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, False]


def test_pandas_numeric_comparator_one_side_not_numeric():
    comparator = PandasNumericComparator()
    assert comparator.compare(pd.Series([1.0, 2.0]), pd.Series(["a", "b"])) is None
    assert comparator.compare(pd.Series(["a", "b"]), pd.Series([1.0, 2.0])) is None
    assert (
        comparator.compare(
            pd.Series([None, None], dtype=object), pd.Series([None, None], dtype=object)
        )
        is None
    )


def test_pandas_numeric_comparator_cast_failure_returns_all_false():
    from decimal import Decimal

    comparator = PandasNumericComparator()
    # Decimal - float raises TypeError, and a signalling NaN cannot be cast to float.
    col1 = pd.Series([Decimal("sNaN"), Decimal("1")], index=[3, 4])
    col2 = pd.Series([1.0, 1.0], index=[3, 4])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [False, False]
    assert result.index.tolist() == [3, 4]
