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

import datetime as dt
from unittest import mock

import pandas as pd
import polars as pl
from datacompy.comparator.string import (
    PandasStringComparator,
    PolarsStringComparator,
    pandas_compare_string_and_date_columns,
    polars_compare_string_and_date_columns,
)


# tests for PolarsStringComparator
def test_polars_string_comparator_exact_match():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c"])
    col2 = pl.Series(["a", "b", "c"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_string_comparator_case_space_insensitivity():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=True, ignore_space=True)
    assert result.to_list() == [True, True, True]

    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=True, ignore_space=False)
    assert result.to_list() == [True, False, False]

    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=False, ignore_space=True)
    assert result.to_list() == [False, True, False]

    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c    "])
    col2 = pl.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=False, ignore_space=False)
    assert result.to_list() == [False, False, False]


def test_polars_string_comparator_none_handling():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", None])
    col2 = pl.Series(["a", "b", None])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, True]


def test_polars_string_comparator_mismatch():
    comparator = PolarsStringComparator()
    col1 = pl.Series(["a", "b", "c"])
    col2 = pl.Series(["a", "b", "d"])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [True, True, False]


def test_polars_string_comparator_error_handling():
    comparator = PolarsStringComparator()
    col1 = pl.Series([1, 2, 3])  # Invalid type for string comparison
    col2 = pl.Series([4, 5, 6])  # Invalid type for string comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pl.Series(["x", "y", "z"])
    col2 = pl.Series(["x", "y", "z", "c"])
    result = comparator.compare(col1, col2)
    assert result is None


def test_polars_string_comparator_temporal_falls_back_to_string_cast():
    """Temporal types that cannot be compared directly are compared as text."""
    comparator = PolarsStringComparator()

    # Time and Date are both temporal but eq_missing rejects the pair, so the
    # comparator casts both to strings, which never agree.
    col1 = pl.Series([dt.time(0, 0), None])
    col2 = pl.Series([dt.date(2023, 1, 1), None])
    result = comparator.compare(col1, col2)
    assert result.dtype == pl.Boolean
    assert result.to_list() == [False, True]

    # A time zone aware datetime never equals a naive one, even at the same
    # wall clock time, because the string forms differ by the zone suffix.
    col1 = pl.Series([dt.datetime(2023, 1, 1), None]).dt.replace_time_zone("UTC")
    col2 = pl.Series([dt.datetime(2023, 1, 1), None])
    result = comparator.compare(col1, col2)
    assert result.to_list() == [False, True]


def test_polars_string_comparator_uncastable_temporal_returns_all_false():
    """When neither direct nor string comparison works, every row is a mismatch."""
    comparator = PolarsStringComparator()
    col1 = pl.Series([dt.timedelta(days=1), dt.timedelta(days=2), None])
    col2 = pl.Series([dt.date(2023, 1, 1), dt.date(2023, 1, 2), None])
    result = comparator.compare(col1, col2)
    assert result.dtype == pl.Boolean
    assert result.to_list() == [False, False, False]


def test_polars_string_comparator_string_and_date():
    comparator = PolarsStringComparator()
    str_col = pl.Series(["2023-01-01", "2023-01-03", None])
    date_col = pl.Series([dt.date(2023, 1, 1), dt.date(2023, 1, 2), None])
    assert comparator.compare(str_col, date_col).to_list() == [True, False, True]
    assert comparator.compare(date_col, str_col).to_list() == [True, False, True]


def test_polars_compare_string_and_date_columns_unparseable_strings():
    """If the date format cannot be inferred, the columns are compared as text."""
    # The first value gives polars nothing to infer a format from, so parsing
    # fails and the date column is cast to its ISO string form instead.
    str_col = pl.Series(["not a date", "2023-01-02", None])
    date_col = pl.Series([dt.date(2023, 1, 1), dt.date(2023, 1, 2), None])
    result = polars_compare_string_and_date_columns(str_col, date_col)
    assert result.to_list() == [False, True, True]

    # The string column is found by dtype, so argument order does not matter.
    result = polars_compare_string_and_date_columns(date_col, str_col)
    assert result.to_list() == [False, True, True]


def test_polars_compare_string_and_date_columns_uncastable_returns_all_false():
    """A column that cannot be cast to a string yields all mismatches."""
    str_col = pl.Series(["a", "b"])
    list_col = pl.Series([[1], [2]])
    result = polars_compare_string_and_date_columns(str_col, list_col)
    assert result.to_list() == [False, False]


# tests for PandasStringComparator
def test_pandas_string_comparator_exact_match():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c"])
    col2 = pd.Series(["a", "b", "c"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_string_comparator_case_space_insensitivity():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=True, ignore_space=True)
    assert result.tolist() == [True, True, True]

    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=True, ignore_space=False)
    assert result.tolist() == [True, False, False]

    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=False, ignore_space=True)
    assert result.tolist() == [False, True, False]

    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c    "])
    col2 = pd.Series(["A", "   b  ", "C"])
    result = comparator.compare(col1, col2, ignore_case=False, ignore_space=False)
    assert result.tolist() == [False, False, False]


def test_pandas_string_comparator_nan_handling():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", float("nan")])
    col2 = pd.Series(["a", "b", float("nan")])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_string_comparator_mismatch():
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b", "c"])
    col2 = pd.Series(["a", "b", "d"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, False]


def test_pandas_string_comparator_error_handling():
    comparator = PandasStringComparator()
    col1 = pd.Series([1, 2, 3])  # Invalid type for string comparison
    col2 = pd.Series([4, 5, 6])  # Invalid type for string comparison
    result = comparator.compare(col1, col2)
    assert result is None

    # different lengths
    col1 = pd.Series(["x", "y", "z"])
    col2 = pd.Series(["x", "y", "z", "c"])
    result = comparator.compare(col1, col2)
    assert result is None


def test_pandas_string_comparator_all_null_object_columns():
    comparator = PandasStringComparator()
    col1 = pd.Series([None, None, None], dtype=object)
    col2 = pd.Series([None, None, None], dtype=object)
    result = comparator.compare(col1, col2)
    assert result.tolist() == [True, True, True]


def test_pandas_string_comparator_all_null_object_vs_string():
    comparator = PandasStringComparator()
    col1 = pd.Series([None, None, None], dtype=object)
    col2 = pd.Series(["a", None, "c"])
    result = comparator.compare(col1, col2)
    assert result.tolist() == [False, True, False]


def test_pandas_compare_string_and_date_columns():
    # Test matching string and date columns
    str_col = pd.Series(["2023-01-01", "2023-02-01", "2023-03-01"])
    date_col = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [True, True, True]

    # Test mismatched values
    str_col = pd.Series(["2023-01-01", "2023-02-02", "2023-03-01"])
    date_col = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [True, False, True]

    # Test null handling
    str_col = pd.Series(["2023-01-01", None, "2023-03-01"])
    date_col = pd.to_datetime(["2023-01-01", pd.NaT, "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [True, True, True]

    # Test mixed format dates
    str_col = pd.Series(["Jan 1 2023", "02-01-2023", "March 1, 2023"])
    date_col = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [True, True, True]

    # Test invalid date strings
    str_col = pd.Series(["not a date", "also not a date", "2023-03-01"])
    date_col = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    result = pandas_compare_string_and_date_columns(str_col, date_col)
    assert result.tolist() == [False, False, True]


def test_pandas_string_comparator_string_fallback_returns_booleans():
    """The last-ditch fallback must return booleans, not the index values."""
    comparator = PandasStringComparator()
    col1 = pd.Series(["a", "b"], index=[7, 9])
    col2 = pd.Series(["a", "b"], index=[7, 9])

    # Force every comparison attempt in the string branch to fail so the
    # fallback is the only path left.
    with (
        mock.patch.object(pd.Series, "fillna", side_effect=Exception("boom")),
        mock.patch.object(pd.Series, "astype", side_effect=Exception("boom")),
    ):
        result = comparator.compare(col1, col2)

    assert result.dtype == bool
    assert result.tolist() == [False, False]
    assert result.index.tolist() == [7, 9]


def test_pandas_string_comparator_mixed_fallback_returns_booleans():
    """Same fallback, reached through the mixed-dtype branch."""
    comparator = PandasStringComparator()
    col1 = pd.Series([1, "a"], index=[7, 9])
    col2 = pd.Series([1, "a"], index=[7, 9])

    with mock.patch.object(pd.Series, "astype", side_effect=Exception("boom")):
        result = comparator.compare(col1, col2)

    assert result.dtype == bool
    assert result.tolist() == [False, False]
    assert result.index.tolist() == [7, 9]


def test_pandas_compare_string_and_date_columns_fallback_returns_booleans():
    """When every conversion fails, each row is reported as a mismatch."""
    str_col = pd.Series(["not a date", "2023-01-02"], index=[3, 5])
    date_col = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02"]), index=[3, 5])

    # The strings defeat both to_datetime attempts; breaking astype removes the
    # string cast, leaving only the all-False fallback.
    with mock.patch.object(pd.Series, "astype", side_effect=Exception("boom")):
        result = pandas_compare_string_and_date_columns(str_col, date_col)

    assert result.dtype == bool
    assert result.tolist() == [False, False]
    assert result.index.tolist() == [3, 5]
