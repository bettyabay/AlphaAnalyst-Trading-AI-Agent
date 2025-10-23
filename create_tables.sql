-- Create tables for AlphaAnalyst Trading AI Agent
-- Run this SQL in your Supabase SQL Editor

-- Create market_data table
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

-- Create indexes for market_data
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON public.market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_date ON public.market_data(date);

-- Create documents table
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

-- Create indexes for documents
CREATE INDEX IF NOT EXISTS idx_documents_symbol ON public.documents(symbol);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON public.documents(filename);

-- Enable Row Level Security (RLS) - optional but recommended
ALTER TABLE public.market_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (adjust as needed for your security requirements)
CREATE POLICY "Enable read access for all users" ON public.market_data FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.market_data FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON public.market_data FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON public.market_data FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON public.documents FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.documents FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON public.documents FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON public.documents FOR DELETE USING (true);
