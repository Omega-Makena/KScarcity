"""Azure data connector stub for adapter-based federation."""


class AzureDataConnectorStub:
    def __init__(self, resource_uri: str):
        self.resource_uri = resource_uri

    def execute_aggregate(self, *args, **kwargs):
        raise NotImplementedError("Azure connector stub: implement adapter for Synapse/Blob/Data Lake/Databricks")
