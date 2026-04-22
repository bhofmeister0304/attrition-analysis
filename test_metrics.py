import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    # 6 employees: overtime=Yes always left, overtime=No never left
    # Leavers earn 3000/4000/5000; stayers earn 5000/6000/7000
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "department":  ["Sales", "Sales", "HR", "HR", "IT", "IT"],
        "overtime":    ["Yes", "No", "Yes", "No", "Yes", "No"],
        "monthly_income": [3000, 5000, 4000, 6000, 5000, 7000],
        "job_satisfaction": [1, 2, 1, 3, 2, 3],
        "attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
    })


# --- attrition_rate ---

def test_attrition_rate_half_leave(sample_df):
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_all_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_none_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values(sample_df):
    result = attrition_by_department(sample_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "department":  ["Sales", "Sales", "HR", "HR", "HR"],
        "attrition":   ["Yes", "Yes", "Yes", "No", "No"],
    })
    result = attrition_by_department(df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_values(sample_df):
    result = attrition_by_overtime(sample_df)

    yes = result[result["overtime"] == "Yes"].iloc[0]
    assert yes["employees"] == 3
    assert yes["leavers"] == 3
    assert yes["attrition_rate"] == 100.0

    no = result[result["overtime"] == "No"].iloc[0]
    assert no["employees"] == 3
    assert no["leavers"] == 0
    assert no["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    # Leavers: 3000, 4000, 5000 -> mean 4000
    leavers = result[result["attrition"] == "Yes"].iloc[0]["avg_monthly_income"]
    assert leavers == 4000.0
    # Stayers: 5000, 6000, 7000 -> mean 6000
    stayers = result[result["attrition"] == "No"].iloc[0]["avg_monthly_income"]
    assert stayers == 6000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_sorted_by_score(sample_df):
    result = satisfaction_summary(sample_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)


def test_satisfaction_summary_rate_is_per_group_not_share_of_leavers():
    # Regression test: attrition_rate must be leavers/employees within each
    # satisfaction tier, not leavers-in-tier/total-leavers-across-all-tiers.
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "job_satisfaction": [1, 1, 2, 2],
        "attrition": ["Yes", "No", "Yes", "Yes"],
    })
    result = satisfaction_summary(df)

    tier1 = result[result["job_satisfaction"] == 1].iloc[0]
    assert tier1["attrition_rate"] == 50.0    # 1 leaver / 2 employees

    tier2 = result[result["job_satisfaction"] == 2].iloc[0]
    assert tier2["attrition_rate"] == 100.0   # 2 leavers / 2 employees
