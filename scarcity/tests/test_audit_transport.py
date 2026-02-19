from scarcity.federation.transport import LoopbackTransport, SimulatedNetworkTransport, TransportConfig, build_transport


def test_build_transport_selects_protocol():
    assert isinstance(build_transport(TransportConfig(protocol="sim")), SimulatedNetworkTransport)
    assert isinstance(build_transport(TransportConfig(protocol="loopback")), LoopbackTransport)
