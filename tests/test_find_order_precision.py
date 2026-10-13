"""Regression test for find_order false positives (HW1 case 10).

At a threshold of 85, "Heavy-Duty Vase" also matched "Heavy-Duty Bar Soap" and
"Heavy-Duty Webcam" because they share a word.
"""

from __future__ import annotations

from agent import db, tools
from agent.auth import AuthContext

SUPPORT = AuthContext(user_id=9501, role="support")


def test_shared_word_does_not_match_a_different_product(world: dict) -> None:
    result = tools.find_order(SUPPORT, "Heavy-Duty Vase")
    assert result["ok"] is True
    assert result["orders"], "expected at least one vase order"
    with db.connection() as conn:
        titles = {p.id: p.title for p in db.list_products(conn)}
    for order in result["orders"]:
        assert titles[order["product_id"]] == "Heavy-Duty Vase", order


def test_single_keyword_query_still_matches_titles_containing_it(world: dict) -> None:
    result = tools.find_order(SUPPORT, "Vase")
    assert result["ok"] is True
    assert result["orders"], "a one-word product query should still find orders"
    with db.connection() as conn:
        titles = {p.id: p.title for p in db.list_products(conn)}
    for order in result["orders"]:
        assert "vase" in titles[order["product_id"]].lower(), order
