"""Test 4: Inference lifecycle, mock model behavior, and latency recording."""

import pytest
from guardslm.models import (
    BaseModel,
    DynaGuardModel,
    MockModel,
    PolicyGuardModel,
    get_model,
)


def test_mock_model_lifecycle():
    model = MockModel(name="mock_test", config={"simulated_latency_ms": 2.0})
    assert not model.is_loaded

    with pytest.raises(RuntimeError):
        model.predict("Prompt")

    model.load()
    assert model.is_loaded

    raw_resp, latency, err = model.predict("TC01: Check access")
    assert raw_resp in ("ALLOW", "BLOCK")
    assert latency > 0.0
    assert err is None

    model.unload()
    assert not model.is_loaded


def test_mock_model_modes():
    # Always allow mode
    m_allow = MockModel(config={"mode": "always_allow"})
    m_allow.load()
    resp, _, _ = m_allow.predict("prompt")
    assert resp == "ALLOW"

    # Always block mode
    m_block = MockModel(config={"mode": "always_block"})
    m_block.load()
    resp, _, _ = m_block.predict("prompt")
    assert resp == "BLOCK"

    # Malformed mode
    m_mal = MockModel(config={"mode": "malformed"})
    m_mal.load()
    resp, _, _ = m_mal.predict("prompt")
    assert "MALFORMED" in resp

    # Error mode
    m_err = MockModel(config={"mode": "error"})
    m_err.load()
    resp, _, err = m_err.predict("prompt")
    assert resp == ""
    assert err is not None
    assert "Simulated mock inference failure" in err


def test_mock_model_target_formats():
    # PolicyGuard format
    m_pg = MockModel(config={"mode": "always_block", "target_format": "policyguard"})
    m_pg.load()
    resp, _, _ = m_pg.predict("prompt")
    assert resp == "violation"

    # DynaGuard format
    m_dg = MockModel(config={"mode": "always_allow", "target_format": "dynaguard"})
    m_dg.load()
    resp, _, _ = m_dg.predict("prompt")
    assert "PASS" in resp


def test_get_model_factory():
    m1 = get_model("mock_guard")
    assert isinstance(m1, MockModel)

    m2 = get_model("policyguard")
    assert isinstance(m2, PolicyGuardModel)
    assert m2.checkpoint == "Rakancorle1/PolicyGuard-4B"

    m3 = get_model("dynaguard")
    assert isinstance(m3, DynaGuardModel)
    assert m3.checkpoint == "tomg-group-umd/DynaGuard-1.7B"

    with pytest.raises(ValueError):
        get_model("unknown_slm")
