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


def test_spark_numeric_comparator_invalid_tolerance_returns_all_false(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame([(1.0, 1.0), (2.0, 2.5)], ["col1", "col2"])
    for tolerance in ("rtol", "atol"):
        kwargs = {tolerance: object()}
        result_col = comparator.compare(
            dataframe=df, col1="col1", col2="col2", **kwargs
        )
        result = df.withColumn("col_match", result_col).select("col_match").collect()
        assert result == [ps.Row(col_match=False), ps.Row(col_match=False)]


def test_spark_numeric_comparator_decimal_vs_double(spark_session):
    from pyspark.sql.types import DecimalType, DoubleType, StructField, StructType

    schema = StructType(
        [
            StructField("col1", DecimalType(10, 2)),
            StructField("col2", DoubleType()),
        ]
    )
    df = spark_session.createDataFrame(
        [(Decimal("1.10"), 1.1), (Decimal("2.20"), 2.25)], schema
    )
    comparator = SparkNumericComparator()
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col).select("col_match").collect()
    assert result == [ps.Row(col_match=True), ps.Row(col_match=False)]


def test_spark_numeric_comparator_null_handling(spark_session):
    from pyspark.sql.types import DoubleType, StructField, StructType

    schema = StructType(
        [
            StructField("col1", DoubleType()),
            StructField("col2", DoubleType()),
        ]
    )
    df = spark_session.createDataFrame(
        [(None, None), (1.0, None), (None, 1.0), (float("nan"), None)], schema
    )
    comparator = SparkNumericComparator()
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col).select("col_match").collect()
    assert result == [
        ps.Row(col_match=True),
        ps.Row(col_match=False),
        ps.Row(col_match=False),
        ps.Row(col_match=False),
    ]


def test_spark_numeric_comparator_tolerance_boundary_inclusive(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame([(1.0, 1.5), (1.0, 1.5000001)], ["col1", "col2"])
    result_col = comparator.compare(
        dataframe=df, col1="col1", col2="col2", rtol=0, atol=0.5
    )
    result = df.withColumn("col_match", result_col).select("col_match").collect()
    assert result == [ps.Row(col_match=True), ps.Row(col_match=False)]


def test_spark_numeric_comparator_relative_tolerance_uses_second_column(spark_session):
    comparator = SparkNumericComparator()
    df = spark_session.createDataFrame([(10.0, 12.0), (12.0, 10.0)], ["col1", "col2"])
    result_col = comparator.compare(
        dataframe=df, col1="col1", col2="col2", rtol=0.18, atol=0
    )
    result = df.withColumn("col_match", result_col).select("col_match").collect()
    assert result == [ps.Row(col_match=True), ps.Row(col_match=False)]


def test_spark_numeric_comparator_empty_dataframe(spark_session):
    from pyspark.sql.types import DoubleType, StructField, StructType

    schema = StructType(
        [
            StructField("col1", DoubleType()),
            StructField("col2", DoubleType()),
        ]
    )
    df = spark_session.createDataFrame([], schema)
    comparator = SparkNumericComparator()
    result_col = comparator.compare(dataframe=df, col1="col1", col2="col2")
    result = df.withColumn("col_match", result_col).select("col_match").collect()
    assert result == []


def test_spark_numeric_comparator_numeric_vs_string_returns_none(spark_session):
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType

    schema = StructType(
        [
            StructField("col1", DoubleType()),
            StructField("col2", StringType()),
        ]
    )
    df = spark_session.createDataFrame([(1.0, "1.0")], schema)
    comparator = SparkNumericComparator()
    assert comparator.compare(dataframe=df, col1="col1", col2="col2") is None
