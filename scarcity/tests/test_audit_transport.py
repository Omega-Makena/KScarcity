from scarcity.federation.transport import LoopbackTransport, SimulatedNetworkTransport, TransportConfig, build_transport
from scarcity.federation.ws_transport import WebSocketTransport


def test_build_transport_selects_protocol():
    assert isinstance(build_transport(TransportConfig(protocol="sim")), SimulatedNetworkTransport)
    assert isinstance(build_transport(TransportConfig(protocol="loopback")), LoopbackTransport)
    assert isinstance(build_transport(TransportConfig(protocol="ws")), WebSocketTransport)
    assert isinstance(build_transport(TransportConfig(protocol="websocket")), WebSocketTransport)
