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

import pytest

pytest.importorskip("pyspark")

import pyspark.sql as ps
from datacompy.comparator.numeric import SparkNumericComparator


# tests for SparkNumericComparator
def test_spark_numeric_comparator_exact_match(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], ["col1", "col2"]
    )
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_approximate_match(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.001), (2.0, 2.002), (3.0, 3.003)], ["col1", "col2"]
    )
    result_col = comparator.compare(
        dataframe=df, col1="col1", col2="col2", rtol=1e-3, atol=1e-3
    )
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_type_casting(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame([(1, 1.0), (2, 2.0), (3, 3.0)], ["col1", "col2"])
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_nan_handling(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (float("nan"), float("nan")), (3.0, 3.0)], ["col1", "col2"]
    )
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=True),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_mismatch(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [(1.0, 1.0), (2.0, 2.5), (3.0, 3.0)], ["col1", "col2"]
    )
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col)
    assert result.select(["col_match"]).collect() == [
        ps.Row(col_match=True),
        ps.Row(col_match=False),
        ps.Row(col_match=True),
    ]


def test_spark_numeric_comparator_error_handling(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(
        [("a", "x"), ("b", "y"), ("c", "z")], ["col1", "col2"]
    )  # Invalid type for numeric comparison
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    assert result_col is None


def test_spark_numeric_comparator_large_integers(spark_session):
    """Large bigint values must not overflow in ANSI mode."""
    comparator = SparkNumericComparator()
    from pyspark.sql.types import LongType, StructField, StructType

    schema = StructType(
        [StructField("col1", LongType()), StructField("col2", LongType())]
    )
    # Use values within double's exact integer range (< 2**53) so comparison is exact.
    # Use rtol=0, atol=0 to test pure equality without tolerance absorbing the diff.
    big = 2**40
    df = spark_session.createDataFrame(
        [(big, big), (big, big - 1000), (big, 0)], schema
    )
    result_col = comparator.compare(
        dataframe=df, col1="col1", col2="col2", rtol=0, atol=0
    )
    result = df.withColumn("col_match", result_col).select("col_match").collect()
    assert result[0][0] is True
    assert result[1][0] is False
    assert result[2][0] is False


def test_spark_numeric_comparator_integer_isnan_safe(spark_session):
    """isnan() must not be called on integer columns (raises in ANSI mode)."""
    comparator = SparkNumericComparator()
    from pyspark.sql.types import IntegerType, StructField, StructType

    schema = StructType(
        [StructField("col1", IntegerType()), StructField("col2", IntegerType())]
    )
    df = spark_session.createDataFrame([(1, 1), (2, 3), (None, 4)], schema)
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col).select("col_match").collect()
    assert result[0][0] is True
    assert result[1][0] is False
    assert result[2][0] is False


def _spark_match(spark_session, rows, schema, **kwargs):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame(rows, schema)
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2", **kwargs)
    if result_col is None:
        return None
    return [row[0] for row in df.select(result_col).collect()]


def test_spark_numeric_comparator_decimal(spark_session):
    from decimal import Decimal

    rows = [
        (Decimal("1.10"), Decimal("1.10")),
        (Decimal("1.10"), Decimal("1.11")),
        (None, None),
        (Decimal("1.00"), None),
    ]
    result = _spark_match(spark_session, rows, "col1 decimal(10,2), col2 decimal(10,2)")
    assert result == [True, False, True, False]


def test_spark_numeric_comparator_decimal_vs_double(spark_session):
    from decimal import Decimal

    rows = [(Decimal("1.10"), 1.1), (Decimal("1.10"), 1.2)]
    result = _spark_match(spark_session, rows, "col1 decimal(10,2), col2 double")
    assert result == [True, False]


def test_spark_numeric_comparator_null_and_nan(spark_session):
    nan = float("nan")
    rows = [
        (None, None),
        (1.0, None),
        (None, 1.0),
        (nan, 1.0),
        (1.0, nan),
        (nan, None),
    ]
    result = _spark_match(spark_session, rows, "col1 double, col2 double")
    assert result == [True, False, False, False, False, False]


def test_spark_numeric_comparator_mixed_types(spark_session):
    assert _spark_match(
        spark_session, [(1.0, 1.0), (2.0, 2.5)], "col1 float, col2 double"
    ) == [True, False]
    assert _spark_match(
        spark_session, [(1, 1), (2, 3)], "col1 tinyint, col2 smallint"
    ) == [True, False]


def test_spark_numeric_comparator_tolerance_boundaries(spark_session):
    rows = [(100.0, 101.0)]
    schema = "col1 double, col2 double"
    assert _spark_match(spark_session, rows, schema, rtol=0.01, atol=0) == [True]
    assert _spark_match(spark_session, rows, schema, rtol=0, atol=1) == [True]
    assert _spark_match(spark_session, rows, schema, rtol=0, atol=0.99) == [False]
    assert _spark_match(spark_session, rows, schema, rtol=0, atol=0) == [False]


def test_spark_numeric_comparator_empty_dataframe(spark_session):
    assert _spark_match(spark_session, [], "col1 double, col2 double") == []


def test_spark_numeric_comparator_one_side_not_numeric(spark_session):
    assert (
        _spark_match(spark_session, [(1.0, "1.0")], "col1 double, col2 string") is None
    )
    assert _spark_match(spark_session, [("1", 1.0)], "col1 string, col2 double") is None
    assert (
        _spark_match(spark_session, [(True, True)], "col1 boolean, col2 boolean")
        is None
    )


def test_spark_numeric_comparator_expression_failure_returns_false(
    spark_session, monkeypatch
):
    import datacompy.comparator.numeric as numeric
    from datacompy.comparator.utility import get_spark_functions

    real_get_spark_functions = get_spark_functions

    class FailingFunctions:
        def __init__(self, psf):
            self._psf = psf

        def isnan(self, *args, **kwargs):
            raise RuntimeError("boom")

        def __getattr__(self, name):
            return getattr(self._psf, name)

    monkeypatch.setattr(
        numeric,
        "get_spark_functions",
        lambda df: FailingFunctions(real_get_spark_functions(df)),
    )
    result = _spark_match(
        spark_session, [(1.0, 1.0), (2.0, 2.0)], "col1 double, col2 double"
    )
    assert result == [False, False]
