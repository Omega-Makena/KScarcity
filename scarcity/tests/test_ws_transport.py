import asyncio
import json
import socket

import pytest

from scarcity.federation.ws_transport import WSTransportConfig, WebSocketTransport


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeInboundWebSocket:
    def __init__(self, messages, remote=("test-peer", 8765)):
        self._messages = iter(messages)
        self.remote_address = remote

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration:
            raise StopAsyncIteration


class _FakePeer:
    def __init__(self, open_state=True, fail_send=False):
        self.open = open_state
        self.fail_send = fail_send
        self.sent_messages = []

    async def send(self, message: str) -> None:
        if self.fail_send:
            raise RuntimeError("simulated send failure")
        self.sent_messages.append(message)


@pytest.mark.asyncio
async def test_send_broadcasts_to_clients_when_no_peers(monkeypatch):
    transport = WebSocketTransport(WSTransportConfig(peer_endpoints=None))

    captured = {}

    async def _capture(message: str) -> None:
        captured["message"] = message

    monkeypatch.setattr(transport, "_broadcast_to_clients", _capture)

    await transport.send("federation.health", {"status": "ok"})

    payload = json.loads(captured["message"])
    assert payload["topic"] == "federation.health"
    assert payload["payload"] == {"status": "ok"}


@pytest.mark.asyncio
async def test_send_to_wraps_and_forwards_message(monkeypatch):
    transport = WebSocketTransport(WSTransportConfig(auth_token="secret"))

    captured = {}

    async def _capture(endpoint: str, message: str) -> None:
        captured["endpoint"] = endpoint
        captured["message"] = message

    monkeypatch.setattr(transport, "_send_to_peer", _capture)

    await transport.send_to("ws://peer:9001", "federation.path_pack", {"n": 1})

    assert captured["endpoint"] == "ws://peer:9001"
    payload = json.loads(captured["message"])
    assert payload["topic"] == "federation.path_pack"
    assert payload["payload"] == {"n": 1}
    assert payload["auth_token"] == "secret"


@pytest.mark.asyncio
async def test_send_to_peer_retries_after_initial_failure(monkeypatch):
    transport = WebSocketTransport(WSTransportConfig())

    failing_peer = _FakePeer(open_state=True, fail_send=True)
    healthy_peer = _FakePeer(open_state=True, fail_send=False)
    calls = {"count": 0}

    async def _ensure_connection(_endpoint: str):
        calls["count"] += 1
        return failing_peer if calls["count"] == 1 else healthy_peer

    monkeypatch.setattr(transport, "_ensure_connection", _ensure_connection)

    await transport._send_to_peer("ws://peer:9002", '{"topic":"x","payload":{}}')

    assert calls["count"] == 2
    assert len(healthy_peer.sent_messages) == 1


@pytest.mark.asyncio
async def test_handle_connection_filters_invalid_json_and_bad_auth():
    transport = WebSocketTransport(WSTransportConfig(auth_token="trusted"))
    received = []

    async def _handler(topic, payload):
        received.append((topic, payload))

    transport.register_handler(_handler)

    ws = _FakeInboundWebSocket(
        [
            "not-json",
            json.dumps({"topic": "ignored", "payload": {"x": 1}, "auth_token": "wrong"}),
            json.dumps({"topic": "accepted", "payload": {"y": 2}, "auth_token": "trusted"}),
        ]
    )

    await transport._handle_connection(ws)

    assert received == [("accepted", {"y": 2})]
    assert transport.connected_clients == 0


@pytest.mark.asyncio
async def test_connected_peers_counts_only_open_connections():
    transport = WebSocketTransport(WSTransportConfig())
    transport._peer_connections = {
        "ws://a:1": _FakePeer(open_state=True),
        "ws://b:2": _FakePeer(open_state=False),
        "ws://c:3": _FakePeer(open_state=True),
    }

    assert transport.connected_peers == 2


@pytest.mark.asyncio
async def test_server_address_reports_host_port():
    config = WSTransportConfig(host="127.0.0.1", port=9009)
    transport = WebSocketTransport(config)
    assert transport.server_address == "ws://127.0.0.1:9009"


@pytest.mark.asyncio
async def test_start_stop_lifecycle_and_client_tracking():
    websockets = pytest.importorskip("websockets")
    port = _find_free_port()
    transport = WebSocketTransport(
        WSTransportConfig(host="127.0.0.1", port=port, ping_interval=1.0, ping_timeout=1.0)
    )

    await transport.start()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}"):
            await asyncio.sleep(0.05)
            assert transport.connected_clients >= 1
    finally:
        await transport.stop()

    assert transport.connected_clients == 0


@pytest.mark.asyncio
async def test_inbound_message_dispatches_to_registered_handler():
    websockets = pytest.importorskip("websockets")
    port = _find_free_port()
    transport = WebSocketTransport(WSTransportConfig(host="127.0.0.1", port=port))
    received = {}
    done = asyncio.Event()

    async def _handler(topic, payload):
        received["topic"] = topic
        received["payload"] = payload
        done.set()

    transport.register_handler(_handler)

    await transport.start()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"topic": "federation.path_pack", "payload": {"k": 7}}))
            await asyncio.wait_for(done.wait(), timeout=2.0)
    finally:
        await transport.stop()

    assert received == {"topic": "federation.path_pack", "payload": {"k": 7}}


@pytest.mark.asyncio
async def test_auth_token_enforced_on_inbound_messages():
    websockets = pytest.importorskip("websockets")
    port = _find_free_port()
    transport = WebSocketTransport(
        WSTransportConfig(host="127.0.0.1", port=port, auth_token="shared-secret")
    )
    received = []

    async def _handler(topic, payload):
        received.append((topic, payload))

    transport.register_handler(_handler)

    await transport.start()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(
                json.dumps(
                    {
                        "topic": "ignored",
                        "payload": {"x": 1},
                        "auth_token": "wrong-secret",
                    }
                )
            )
            await asyncio.sleep(0.05)
            assert received == []

            await ws.send(
                json.dumps(
                    {
                        "topic": "accepted",
                        "payload": {"x": 2},
                        "auth_token": "shared-secret",
                    }
                )
            )
            await asyncio.wait_for(_wait_until(lambda: len(received) == 1), timeout=2.0)
    finally:
        await transport.stop()

    assert received == [("accepted", {"x": 2})]


async def _wait_until(predicate, interval: float = 0.01):
    while not predicate():
        await asyncio.sleep(interval)
