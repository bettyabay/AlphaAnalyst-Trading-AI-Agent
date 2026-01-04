from tradingagents.dataflows.ingestion_pipeline import DataIngestionPipeline
from datetime import datetime, timedelta

def main():
    pipeline = DataIngestionPipeline()
    
    # Configuration
    symbol = "C:XAUUSD"  # Polygon symbol for Gold (Spot)
    target_timezone = "Asia/Dubai" # GMT+4
    years = 2
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * years)
    
    print(f"Starting ingestion for {symbol} from {start_date.date()} to {end_date.date()}")
    print(f"Target Timezone: {target_timezone}")
    
    # Ingest 1-minute data
    # Set resume_from_latest=False for fresh ingestion after manual deletion
    success = pipeline.ingest_historical_data(
        symbol=symbol,
        interval="1min",
        start_date=start_date,
        end_date=end_date,
        target_timezone=target_timezone,
        resume_from_latest=True  # False = start from beginning (for fresh ingestion)
    )
    
    if success:
        print("Ingestion completed successfully.")
    else:
        print("Ingestion failed.")

if __name__ == "__main__":
    main()
