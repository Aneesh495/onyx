"""
Onyx Python SDK
High-throughput GPU-accelerated columnar query execution and streaming kernel.
"""

from .client import OnyxClient
from .dataframe import OnyxDataFrame
from .types import DataType

DataFrame = OnyxDataFrame
Client = OnyxClient

__version__ = "0.1.0"
__all__ = ["OnyxClient", "OnyxDataFrame", "DataType", "DataFrame", "Client"]
