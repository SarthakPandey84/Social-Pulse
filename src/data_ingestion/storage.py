"""Storage manager for raw lake landing, processed datasets, and SQLite persistence.

Supports historical snapshot retention, idempotent ingestion, and latest-state SQL views.
"""

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Optional, Tuple
import pandas as pd
from utils.helpers import get_project_root


class DataStorageManager:
    """Manages raw immutable landing files, processed data artifacts, and SQLite database storage."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        db_path: Optional[Path] = None
    ):
        root = get_project_root()
        self.raw_dir = raw_dir or (root / "data" / "raw")
        self.processed_dir = processed_dir or (root / "data" / "processed")
        self.db_path = db_path or (root / "database" / "social_analytics.db")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def save_raw_snapshot(self, df: pd.DataFrame, platform: str) -> Path:
        """Write immutable timestamped raw ingestion snapshot to data/raw/."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        target_path = self.raw_dir / f"raw_{platform}_{timestamp_str}.parquet"
        df.to_parquet(target_path, index=False)
        return target_path

    def save_processed_dataset(self, df: pd.DataFrame, base_name: str = "processed_posts") -> Tuple[Path, Path]:
        """Save cleaned, feature-engineered dataset to data/processed/ in Parquet and CSV formats."""
        parquet_path = self.processed_dir / f"{base_name}.parquet"
        csv_path = self.processed_dir / f"{base_name}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        return parquet_path, csv_path

    def persist_to_sqlite(self, df: pd.DataFrame, table_name: str = "posts", if_exists: str = "replace") -> int:
        """Persist DataFrame into SQLite database, indexing snapshot tuples and creating a latest-record view."""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            # Cast timestamp columns to string for SQLite compatibility
            df_sql = df.copy()
            for col in df_sql.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
                df_sql[col] = df_sql[col].astype(str)

            df_sql.to_sql(table_name, conn, if_exists=if_exists, index=False)

            cursor = conn.cursor()
            # Index on full snapshot identity
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_snapshot ON {table_name}(platform, post_id, collection_timestamp)"
            )

            # Create or update latest-record view
            view_sql = f"""
            CREATE VIEW IF NOT EXISTS latest_{table_name} AS
            SELECT p.*
            FROM {table_name} p
            INNER JOIN (
                SELECT platform, post_id, MAX(collection_timestamp) AS max_cts
                FROM {table_name}
                GROUP BY platform, post_id
            ) latest
            ON p.platform = latest.platform 
            AND p.post_id = latest.post_id 
            AND p.collection_timestamp = latest.max_cts;
            """
            cursor.execute(view_sql)

        return len(df)

    def load_from_sqlite(self, table_name: str = "posts") -> pd.DataFrame:
        """Read all historical records from SQLite database."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return df

    def load_latest_from_sqlite(self, table_name: str = "posts") -> pd.DataFrame:
        """Read only the latest observation per post using the latest-record SQL view."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            df = pd.read_sql_query(f"SELECT * FROM latest_{table_name}", conn)
        return df

    def load_history_for_post(self, platform: str, post_id: str, table_name: str = "posts") -> pd.DataFrame:
        """Query longitudinal history of metric progression for an individual post."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            query = f"""
            SELECT * FROM {table_name}
            WHERE platform = ? AND post_id = ?
            ORDER BY collection_timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(platform, post_id))
        return df
