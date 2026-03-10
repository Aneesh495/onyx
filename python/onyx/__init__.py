"""
Onyx Python SDK
High-throughput GPU-accelerated columnar query execution and streaming kernel.
"""

from .client import OnyxClient
from .dataframe import OnyxDataFrame
from .streaming import OnyxStreamProducer
from .types import DataType

DataFrame = OnyxDataFrame
Client = OnyxClient
StreamProducer = OnyxStreamProducer

__version__ = "0.1.0"
__all__ = ["OnyxClient", "OnyxDataFrame", "OnyxStreamProducer", "DataType", "DataFrame", "Client", "StreamProducer"]
