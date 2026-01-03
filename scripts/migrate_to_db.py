"""
Migration script to move historical snapshots from JSON files to PostgreSQL.
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

from market_liquidity_monitor.data_engine.database import db_manager
from market_liquidity_monitor.data_engine.models import HistoricalSnapshot

async def migrate_json_to_db(storage_dir: str = "./data/historical"):
    """
    Read all JSON snapshot files and insert them into PostgreSQL.
    """
    await db_manager.connect()
    if not db_manager.is_active:
        print("❌ Database is not active. Aborting migration.")
        return

    storage_path = Path(storage_dir)
    if not storage_path.exists():
        print(f"⚠️ Storage directory {storage_dir} not found.")
        return

    json_files = list(storage_path.glob("*.json"))
    print(f"🔍 Found {len(json_files)} JSON files to migrate.")

    total_migrated = 0
    for file_path in json_files:
        print(f"📂 Processing {file_path.name}...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                snapshots = [HistoricalSnapshot(**s) for s in data]
            
            for snapshot in snapshots:
                await db_manager.store_snapshot(snapshot.model_dump())
                total_migrated += 1
            
            print(f"✅ Migrated {len(snapshots)} snapshots from {file_path.name}")
        except Exception as e:
            print(f"❌ Error migrating {file_path.name}: {e}")

    await db_manager.disconnect()
    print(f"\n✨ Migration complete. Total snapshots migrated: {total_migrated}")

if __name__ == "__main__":
    # Ensure CWD is project root
    asyncio.run(migrate_json_to_db())
