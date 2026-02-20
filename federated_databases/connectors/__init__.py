"""Connector implementations for federated data execution."""

from .base import ConnectorSpec
from .sqlite_node import SQLiteNodeConnector
from .postgres_node import PostgresNodeConnector
from .oracle_stub import OracleConnectorStub
from .sqlserver_stub import SQLServerConnectorStub
from .azure_stub import AzureDataConnectorStub
from .http_api_stub import HTTPAPIConnectorStub
from .registry import ConnectorFactory

__all__ = [
    "ConnectorSpec",
    "SQLiteNodeConnector",
    "PostgresNodeConnector",
    "OracleConnectorStub",
    "SQLServerConnectorStub",
    "AzureDataConnectorStub",
    "HTTPAPIConnectorStub",
    "ConnectorFactory",
]
