"""Oracle connector stub for adapter-based federation."""


class OracleConnectorStub:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def execute_aggregate(self, *args, **kwargs):
        raise NotImplementedError("Oracle connector stub: implement adapter binding to your Oracle client")
