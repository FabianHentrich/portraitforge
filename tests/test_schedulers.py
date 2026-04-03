"""Tests für pipeline/utils/schedulers.py — SCHEDULERS dict."""
import pytest
from pipeline.utils.schedulers import SCHEDULERS

EXPECTED_KEYS = {"euler", "dpm++2m", "dpm++2m_karras", "euler_a", "ddim"}


def test_all_expected_keys_present():
    assert EXPECTED_KEYS == set(SCHEDULERS.keys())


def test_all_values_are_callable():
    for key, factory in SCHEDULERS.items():
        assert callable(factory), f"SCHEDULERS['{key}'] is not callable"


def test_no_extra_keys():
    assert set(SCHEDULERS.keys()) == EXPECTED_KEYS


def test_schedulers_is_dict():
    assert isinstance(SCHEDULERS, dict)
