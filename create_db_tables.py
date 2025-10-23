#!/usr/bin/env python3
"""
Simple script to create database tables in Supabase
Run this script to create the necessary tables for the AlphaAnalyst Trading AI Agent
"""
import os
from dotenv import load_dotenv
from tradingagents.database.config import get_supabase

def create_tables():
    """Create database tables"""
    print("Creating database tables...")
    
    # Load environment variables
    load_dotenv()
    
    # Get Supabase connection
    supabase = get_supabase()
    if not supabase:
        print("Error: Could not connect to Supabase. Please check your .env file.")
        return False
    
    try:
        # Test if tables already exist
        print("Checking if tables exist...")
        
        # Test market_data table
        try:
            supabase.table("market_data").select("id").limit(1).execute()
            print("market_data table already exists")
        except Exception as e:
            print(f"market_data table does not exist: {e}")
            print("You need to create the tables manually in Supabase dashboard.")
            print("Please run the SQL in create_tables.sql in your Supabase SQL Editor.")
            return False
        
        # Test documents table
        try:
            supabase.table("documents").select("id").limit(1).execute()
            print("documents table already exists")
        except Exception as e:
            print(f"documents table does not exist: {e}")
            print("You need to create the tables manually in Supabase dashboard.")
            print("Please run the SQL in create_tables.sql in your Supabase SQL Editor.")
            return False
        
        print("All tables exist and are accessible!")
        return True
        
    except Exception as e:
        print(f"Error checking tables: {e}")
        return False

if __name__ == "__main__":
    print("AlphaAnalyst Trading AI Agent - Database Table Checker")
    print("=" * 50)
    
    if create_tables():
        print("\nDatabase tables are ready!")
        print("You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("\nPlease create the tables manually using the SQL in create_tables.sql")
        print("Go to your Supabase dashboard -> SQL Editor and run the SQL script.")
