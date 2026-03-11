"""Canonical  feature reference for the German Credit dataset."""
from __future__ import annotations

from typing import Any

import pandas as pd


FEATURE_REFERENCE: list[dict[str, str]] = [
    {
        "name": "checking_status",
        "type": "categorical",
        "description": "Status of existing checking account (DM balance).",
    },
    {"name": "duration", "type": "numeric", "description": "Loan duration in months."},
    {
        "name": "credit_history",
        "type": "categorical",
        "description": "History of past credits and repayment behavior.",
    },
    {
        "name": "purpose",
        "type": "categorical",
        "description": "Purpose of the loan (car, furniture, education, etc.).",
    },
    {
        "name": "credit_amount",
        "type": "numeric",
        "description": "Loan amount in Deutsche Mark.",
    },
    {
        "name": "savings_status",
        "type": "categorical",
        "description": "Savings account / bonds balance.",
    },
    {
        "name": "employment",
        "type": "categorical",
        "description": "Years of present employment.",
    },
    {
        "name": "installment_commitment",
        "type": "numeric",
        "description": "Installment rate as % of disposable income.",
    },
    {
        "name": "personal_status",
        "type": "categorical",
        "description": "Personal status and sex.",
    },
    {
        "name": "other_parties",
        "type": "categorical",
        "description": "Other debtors or guarantors.",
    },
    {
        "name": "residence_since",
        "type": "numeric",
        "description": "Years at present residence.",
    },
    {
        "name": "property_magnitude",
        "type": "categorical",
        "description": "Most valuable available property.",
    },
    {"name": "age", "type": "numeric", "description": "Age in years."},
    {
        "name": "other_payment_plans",
        "type": "categorical",
        "description": "Other installment plans (bank, stores, none).",
    },
    {
        "name": "housing",
        "type": "categorical",
        "description": "Housing situation (own, free, rent).",
    },
    {
        "name": "existing_credits",
        "type": "numeric",
        "description": "Number of existing credits at this bank.",
    },
    {
        "name": "job",
        "type": "categorical",
        "description": "Job type / skill level.",
    },
    {
        "name": "num_dependents",
        "type": "numeric",
        "description": "Number of people liable to provide maintenance for.",
    },
    {
        "name": "own_telephone",
        "type": "categorical",
        "description": "Whether a telephone is registered in applicant's name.",
    },
    {
        "name": "foreign_worker",
        "type": "categorical",
        "description": "Whether the applicant is a foreign worker.",
    },
]

TARGET_REFERENCE: dict[str, str] = {
    "name": "class",
    "type": "binary",
    "description": "Credit risk: 1=Good, 2=Bad. Remap to 0=Good, 1=Bad.",
}


def get_feature_names() -> list[str]:
    """Return canonical base feature names in source-order."""
    return [item["name"] for item in FEATURE_REFERENCE]


def get_numeric_feature_names() -> list[str]:
    """Return canonical numeric base feature names."""
    return [item["name"] for item in FEATURE_REFERENCE if item["type"] == "numeric"]


def get_categorical_feature_names() -> list[str]:
    """Return canonical categorical base feature names."""
    return [item["name"] for item in FEATURE_REFERENCE if item["type"] == "categorical"]


def validate_config_schema(config: dict[str, Any]) -> None:
    """Validate config schema fields against canonical feature reference."""
    configured_columns = config["data"]["column_names"]
    expected_columns = get_feature_names() + [TARGET_REFERENCE["name"]]
    if configured_columns != expected_columns:
        raise ValueError(
            "data.column_names does not match canonical feature order. "
            f"Expected {expected_columns}, got {configured_columns}."
        )

    configured_numeric = sorted(config["schema"]["numeric_columns"])
    configured_categorical = sorted(config["schema"]["categorical_columns"])
    expected_numeric = sorted(get_numeric_feature_names())
    expected_categorical = sorted(get_categorical_feature_names())

    if configured_numeric != expected_numeric:
        raise ValueError(
            "schema.numeric_columns does not match canonical numeric features. "
            f"Expected {expected_numeric}, got {configured_numeric}."
        )
    if configured_categorical != expected_categorical:
        raise ValueError(
            "schema.categorical_columns does not match canonical categorical features. "
            f"Expected {expected_categorical}, got {configured_categorical}."
        )

    if config["schema"]["target_column"] != TARGET_REFERENCE["name"]:
        raise ValueError(
            "schema.target_column mismatch. "
            f"Expected '{TARGET_REFERENCE['name']}', got '{config['schema']['target_column']}'."
        )


def validate_dataframe_features(df: pd.DataFrame, target_column: str) -> None:
    """Validate dataframe has exact canonical feature set + target."""
    expected_columns = get_feature_names() + [target_column]
    if list(df.columns) != expected_columns:
        raise ValueError(
            "Unexpected dataframe columns/order. "
            f"Expected {expected_columns}, got {list(df.columns)}."
        )

