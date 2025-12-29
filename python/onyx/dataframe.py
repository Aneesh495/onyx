from typing import List, Optional, Dict, Any

class OnyxDataFrame:
    """
    Lazy DataFrame abstraction compiled down to Onyx SQL expressions.
    """

    def __init__(self, table_name: str, client: Optional[Any] = None):
        self.table_name = table_name
        self.client = client
        self._select_cols: List[str] = ["*"]
        self._where_clause: Optional[str] = None
        self._group_by: List[str] = []
        self._limit_count: Optional[int] = None

    def select(self, *columns: str) -> "OnyxDataFrame":
        df = self._clone()
        df._select_cols = list(columns)
        return df

    def filter(self, condition: str) -> "OnyxDataFrame":
        df = self._clone()
        df._where_clause = condition
        return df

    def limit(self, n: int) -> "OnyxDataFrame":
        df = self._clone()
        df._limit_count = n
        return df

    def to_sql(self) -> str:
        cols = ", ".join(self._select_cols)
        sql = f"SELECT {cols} FROM {self.table_name}"
        if self._where_clause:
            sql += f" WHERE {self._where_clause}"
        if self._limit_count is not None:
            sql += f" LIMIT {self._limit_count}"
        return sql + ";"

    def collect(self) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("No OnyxClient attached to execute DataFrame")
        return self.client.query(self.to_sql())

    def _clone(self) -> "OnyxDataFrame":
        df = OnyxDataFrame(self.table_name, self.client)
        df._select_cols = list(self._select_cols)
        df._where_clause = self._where_clause
        df._group_by = list(self._group_by)
        df._limit_count = self._limit_count
        return df
