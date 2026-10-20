"""Authentication tests for the HW2 session endpoints.

Offline: no Langfuse, Docker, or model provider key.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server import app as server_app


def test_session_creation_rejects_a_role_that_differs_from_the_database(world: dict) -> None:
    server_app._SESSIONS.clear()
    # User 1 is a shopper in the database; claiming "support" must be refused.
    with pytest.raises(HTTPException) as exc:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="support"))
    assert exc.value.status_code == 403
    assert server_app._SESSIONS == {}  # no session or token was created


def test_token_for_one_session_cannot_authorize_another(world: dict) -> None:
    server_app._SESSIONS.clear()
    first = server_app.create_session(server_app.SessionCreate(user_id=1, role="shopper"))
    second = server_app.create_session(server_app.SessionCreate(user_id=1, role="shopper"))

    # The token works for its own session...
    ctx = server_app._authorize(first["session_id"], f"Bearer {first['token']}")
    assert ctx.user_id == 1
    # ...but not for a different one, even for the same user.
    with pytest.raises(HTTPException) as exc:
        server_app._authorize(second["session_id"], f"Bearer {first['token']}")
    assert exc.value.status_code == 403
