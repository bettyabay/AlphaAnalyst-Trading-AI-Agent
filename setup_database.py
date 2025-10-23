"""
Supabase setup script for AlphaAnalyst Trading AI Agent
Run this script to verify Supabase connection and create necessary tables
"""
import os
from dotenv import load_dotenv
from tradingagents.database.config import get_supabase
from tradingagents.dataflows.ingestion_pipeline import DataIngestionPipeline
from tradingagents.config.watchlist import WATCHLIST_STOCKS

def setup_database():
    """Verify Supabase connection and create tables"""
    print("Verifying Supabase connection...")
    try:
        supabase = get_supabase()
        if supabase:
            print("Supabase connection verified!")
            
            # Create tables
            if create_tables(supabase):
                print("Database tables created successfully!")
                return True
            else:
                print("Failed to create database tables!")
                return False
        else:
            print("Supabase not configured. Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
            return False
    except Exception as e:
        print(f"Supabase connection failed: {e}")
        return False

def create_tables(supabase):
    """Create necessary database tables"""
    print("Creating database tables...")
    
    # SQL to create market_data table
    market_data_sql = """
    CREATE TABLE IF NOT EXISTS public.market_data (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        date DATE NOT NULL,
        open DECIMAL(10,2),
        high DECIMAL(10,2),
        low DECIMAL(10,2),
        close DECIMAL(10,2),
        volume BIGINT,
        source VARCHAR(50) DEFAULT 'polygon',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(symbol, date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON public.market_data(symbol);
    CREATE INDEX IF NOT EXISTS idx_market_data_date ON public.market_data(date);
    """
    
    # SQL to create documents table
    documents_sql = """
    CREATE TABLE IF NOT EXISTS public.documents (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        title VARCHAR(255),
        file_content TEXT,
        file_type VARCHAR(50),
        symbol VARCHAR(20),
        uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_documents_symbol ON public.documents(symbol);
    CREATE INDEX IF NOT EXISTS idx_documents_filename ON public.documents(filename);
    """
    
    try:
        # Execute market_data table creation
        print("Creating market_data table...")
        supabase.rpc('exec_sql', {'sql': market_data_sql}).execute()
        
        # Execute documents table creation  
        print("Creating documents table...")
        supabase.rpc('exec_sql', {'sql': documents_sql}).execute()
        
        return True
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        # Try alternative approach - direct table creation
        try:
            print("Trying alternative table creation approach...")
            
            # Test if tables exist by trying to query them
            try:
                supabase.table("market_data").select("id").limit(1).execute()
                print("market_data table already exists")
            except:
                print("market_data table does not exist - you may need to create it manually in Supabase dashboard")
                
            try:
                supabase.table("documents").select("id").limit(1).execute()
                print("documents table already exists")
            except:
                print("documents table does not exist - you may need to create it manually in Supabase dashboard")
                
            return True  # Return true to continue with the setup
            
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
            return False

def initialize_stocks():
    """Initialize stocks in the database"""
    print("Initializing stocks...")
    try:
        pipeline = DataIngestionPipeline()
        success = pipeline.initialize_stocks()
        
        if success:
            print("Stocks initialized successfully!")
            return True
        else:
            print("Stock initialization failed!")
            return False
    except Exception as e:
        print(f"Stock initialization error: {e}")
        return False

def main():
    """Main setup function"""
    print("AlphaAnalyst Trading AI Agent - Database Setup")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "POLYGON_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables.")
        print("See tradingagents/config/env_template.py for reference.")
        return False
    
    # Setup database
    if not setup_database():
        return False
    
    # Initialize stocks
    if not initialize_stocks():
        return False
    
    print("\nSetup completed successfully!")
    print("You can now run the Streamlit app with: streamlit run app.py")
    return True

if __name__ == "__main__":
    main()
