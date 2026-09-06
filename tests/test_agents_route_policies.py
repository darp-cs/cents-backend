from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from app.routes.agents import _apply_enabled_policy, _should_auto_enable_new_version


def _record(version: int, enabled: bool, is_valid: bool = True) -> SimpleNamespace:
    return SimpleNamespace(version=version, enabled=enabled, is_valid=is_valid)


def test_enable_target_disables_other_versions() -> None:
    v1 = _record(version=1, enabled=True, is_valid=True)
    v2 = _record(version=2, enabled=False, is_valid=True)
    records = [v2, v1]

    _apply_enabled_policy(records, v2, desired_enabled=True)

    assert v2.enabled is True
    assert v1.enabled is False


def test_enable_invalid_version_is_rejected() -> None:
    v1 = _record(version=1, enabled=True, is_valid=True)
    v2 = _record(version=2, enabled=False, is_valid=False)

    with pytest.raises(HTTPException) as exc:
        _apply_enabled_policy([v2, v1], v2, desired_enabled=True)

    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc.value.detail == "Invalid templates cannot be enabled."


def test_disabling_last_enabled_version_is_rejected() -> None:
    v1 = _record(version=1, enabled=True, is_valid=True)
    v2 = _record(version=2, enabled=False, is_valid=True)

    with pytest.raises(HTTPException) as exc:
        _apply_enabled_policy([v2, v1], v1, desired_enabled=False)

    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc.value.detail == "At least one version must remain enabled for this agent."


def test_disabling_enabled_version_with_alternative_active_is_allowed() -> None:
    v1 = _record(version=1, enabled=True, is_valid=True)
    v2 = _record(version=2, enabled=True, is_valid=True)

    _apply_enabled_policy([v2, v1], v1, desired_enabled=False)

    assert v1.enabled is False
    assert v2.enabled is True


def test_auto_enable_new_valid_version_when_none_enabled() -> None:
    assert _should_auto_enable_new_version(existing_enabled_count=0, is_valid=True) is True


def test_do_not_auto_enable_when_existing_enabled_version_exists() -> None:
    assert _should_auto_enable_new_version(existing_enabled_count=1, is_valid=True) is False


def test_do_not_auto_enable_invalid_template() -> None:
    assert _should_auto_enable_new_version(existing_enabled_count=0, is_valid=False) is False
