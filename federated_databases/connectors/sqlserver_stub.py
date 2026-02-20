"""SQL Server connector stub for adapter-based federation."""


class SQLServerConnectorStub:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def execute_aggregate(self, *args, **kwargs):
        raise NotImplementedError("SQL Server connector stub: implement adapter binding to your pyodbc/sqlalchemy client")
