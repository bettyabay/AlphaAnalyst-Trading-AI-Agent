"""
Quick script to verify what index symbols are stored in the database.
Run this to check if your indices are stored with the correct symbols (I:SPX, I:NDX, I:DJI).
"""
from ingest_indices_polygon import verify_indices_in_database

if __name__ == "__main__":
    print("🔍 Verifying index symbols in database...\n")
    verify_indices_in_database()

