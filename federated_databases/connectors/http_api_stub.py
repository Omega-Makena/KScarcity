"""HTTP API connector stub for adapter-based federation."""


class HTTPAPIConnectorStub:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def execute_aggregate(self, *args, **kwargs):
        raise NotImplementedError("HTTP API connector stub: implement adapter for your approved API contract")
