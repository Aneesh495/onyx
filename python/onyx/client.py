import json
import urllib.request
import urllib.error
from typing import Dict, Any, List

class OnyxClient:
    """
    Client for interacting with the Onyx GPU query engine server.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.base_url = f"http://{host}:{port}"

    def query(self, sql: str) -> Dict[str, Any]:
        """
        Execute a SQL query against the Onyx GPU query engine.
        """
        url = f"{self.base_url}/query"
        payload = json.dumps({"query": sql}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return {"status": "Error", "message": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Fetch real-time GPU memory and execution engine telemetry.
        """
        url = f"{self.base_url}/metrics"
        try:
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return {"status": "Error", "message": str(e)}

    def list_tables(self) -> List[Dict[str, Any]]:
        """
        Fetch catalog tables currently in memory.
        """
        url = f"{self.base_url}/tables"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("tables", [])
        except urllib.error.URLError:
            return []
