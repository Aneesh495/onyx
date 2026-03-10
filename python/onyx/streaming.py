import json
import socket
from typing import Dict, Any, Generator, Optional

class OnyxStreamProducer:
    """
    High-throughput socket-based streaming ingestion producer for Onyx.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9091):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def emit_event(self, table_name: str, record: Dict[str, Any]) -> None:
        payload = json.dumps({"table": table_name, "record": record}) + "\n"
        if self.sock:
            self.sock.sendall(payload.encode("utf-8"))

    def close(self) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None
