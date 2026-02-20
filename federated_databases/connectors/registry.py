"""Connector factory for heterogeneous federated adapters."""

from __future__ import annotations

from typing import Any, Dict

from .sqlite_node import SQLiteNodeConnector
from .postgres_node import PostgresNodeConnector
from .oracle_stub import OracleConnectorStub
from .sqlserver_stub import SQLServerConnectorStub
from .azure_stub import AzureDataConnectorStub
from .http_api_stub import HTTPAPIConnectorStub


class ConnectorFactory:
    """Instantiate connectors by source type without coupling the control plane."""

    @staticmethod
    def create(spec: Dict[str, Any]):
        source_type = str(spec.get("source_type", "")).lower()
        location = str(spec.get("location", ""))

        if source_type == "sqlite":
            return SQLiteNodeConnector(location)
        if source_type == "postgres":
            return PostgresNodeConnector(location)
        if source_type == "oracle":
            return OracleConnectorStub(location)
        if source_type in {"sqlserver", "mssql"}:
            return SQLServerConnectorStub(location)
        if source_type in {"azure", "azure_store", "azure_storage"}:
            return AzureDataConnectorStub(location)
        if source_type in {"http", "http_api", "api"}:
            return HTTPAPIConnectorStub(location)

        raise ValueError(f"Unsupported source_type: {source_type}")
