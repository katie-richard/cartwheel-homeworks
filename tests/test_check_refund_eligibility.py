"""Tests for the HW1 additional tool, check_refund_eligibility."""

from __future__ import annotations

import pytest

from agent import db, tools
from agent.agent import TOOLS_BY_ROLE
from agent.auth import AuthContext
from agent.config import load_facts
from seed.eligibility import effective_return_window_days, is_refund_eligible

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
MERCHANT_STORE_2 = AuthContext(user_id=9002, role="merchant", store_id=2)
MERCHANT_STORE_1 = AuthContext(user_id=9001, role="merchant", store_id=1)
SUPPORT = AuthContext(user_id=9501, role="support")


def test_eligible_order_reports_window_and_days_remaining(world: dict) -> None:
    result = tools.check_refund_eligibility(SHOPPER_1, 4127)
    assert result["ok"] is True
    assert result["eligible"] is True
    assert result["reason_code"] == "eligible"
    assert result["return_window_days"] == 30
    assert result["days_since_delivery"] == 12
    assert result["days_remaining"] == 18
    assert result["days_past_window"] is None
    assert result["policy_ids"] == ["cw-returns"]


def test_expired_order_reports_days_past_window(world: dict) -> None:
    result = tools.check_refund_eligibility(SHOPPER_1, 3980)
    assert result["ok"] is True
    assert result["eligible"] is False
    assert result["reason_code"] == "window_expired"
    assert result["days_since_delivery"] == 45
    assert result["days_past_window"] == 15
    assert result["days_remaining"] is None


def test_scope_is_checked_before_order_details(world: dict) -> None:
    denied = tools.check_refund_eligibility(MERCHANT_STORE_2, 4127)
    assert denied["ok"] is False
    assert denied["error"] == "permission_denied"
    assert "delivered" not in denied["reason"]
    assert tools.check_refund_eligibility(MERCHANT_STORE_1, 4127)["ok"] is True
    assert tools.check_refund_eligibility(SUPPORT, 4127)["ok"] is True


def test_unknown_order_is_not_found(world: dict) -> None:
    result = tools.check_refund_eligibility(SUPPORT, 99999999)
    assert result["ok"] is False
    assert result["error"] == "not_found"


def test_store_override_window_and_policy_id(world: dict) -> None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT o.id FROM orders o JOIN stores s ON s.id = o.store_id "
            "WHERE s.slug = 'meridian-cycles' AND o.status = 'delivered' LIMIT 1"
        ).fetchone()
    result = tools.check_refund_eligibility(SUPPORT, row[0])
    assert result["ok"] is True
    assert result["return_window_days"] == 21
    assert result["policy_ids"] == ["cw-returns", "store-meridian-cycles-policy"]


def test_non_delivered_order_is_not_eligible(world: dict) -> None:
    with db.connection() as conn:
        row = conn.execute("SELECT id FROM orders WHERE status = 'shipped' LIMIT 1").fetchone()
    result = tools.check_refund_eligibility(SUPPORT, row[0])
    assert result["eligible"] is False
    assert result["reason_code"] == "not_delivered"


def test_eligible_flag_agrees_with_the_eligibility_oracle(world: dict) -> None:
    platform_window = load_facts()["return_window_days"]
    with db.connection() as conn:
        as_of = db.world_asof(conn)
        orders = db.list_order_search_candidates(conn, all_orders=True)[:500]
        for order in orders:
            store = db.get_store(conn, order.store_id)
            window = effective_return_window_days(
                platform_window, store.return_window_days_override
            )
            expected = is_refund_eligible(
                status=order.status,
                delivered_at=order.delivered_at,
                as_of=as_of,
                return_window_days=window,
            )
            result = tools.check_refund_eligibility(SUPPORT, order.id)
            assert result["eligible"] is expected, order.id


@pytest.mark.parametrize("role", ["shopper", "merchant", "support"])
def test_tool_is_registered_for_every_role(role: str) -> None:
    names = {getattr(tool, "name", None) for tool in TOOLS_BY_ROLE[role]}
    assert "check_refund_eligibility" in names
