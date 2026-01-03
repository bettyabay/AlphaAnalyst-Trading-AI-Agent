import sys
import os
import inspect

# Add project root to path
sys.path.append(os.getcwd())

print("Testing imports...")

try:
    from tradingagents.dataflows.universal_ingestion import ingest_from_polygon_api
    print("Successfully imported ingest_from_polygon_api from universal_ingestion")
    
    # Check signature
    sig = inspect.signature(ingest_from_polygon_api)
    print(f"Signature of ingest_from_polygon_api: {sig}")
    
except ImportError as e:
    print(f"Failed to import ingest_from_polygon_api: {e}")
    sys.exit(1)

try:
    from tradingagents.dataflows.gold_ingestion import ingest_gold_from_api
    print("Successfully imported ingest_gold_from_api from gold_ingestion")
    
    # Check signature
    sig = inspect.signature(ingest_gold_from_api)
    print(f"Signature of ingest_gold_from_api: {sig}")
    
except ImportError as e:
    print(f"Failed to import ingest_gold_from_api: {e}")
    sys.exit(1)

try:
    # Check ingest_gold_gmt4.py (it's a script, so we can't easily import it without running it, 
    # but we can check if it imports DataIngestionPipeline correctly)
    from tradingagents.dataflows.ingestion_pipeline import DataIngestionPipeline
    print("Successfully imported DataIngestionPipeline")
except ImportError as e:
    print(f"Failed to import DataIngestionPipeline: {e}")
    sys.exit(1)

print("All import tests passed.")
