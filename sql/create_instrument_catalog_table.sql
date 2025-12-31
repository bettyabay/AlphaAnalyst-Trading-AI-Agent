-- Create table for instrument catalog (built-in and custom instruments)
-- Stores instrument metadata and category mapping

CREATE TABLE IF NOT EXISTS instrument_catalog (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    name VARCHAR(200),
    category VARCHAR(50) NOT NULL,
    sector VARCHAR(100),
    exchange VARCHAR(50),
    source VARCHAR(50) DEFAULT 'excel_upload',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, category)
);

CREATE INDEX IF NOT EXISTS idx_instrument_catalog_symbol 
    ON instrument_catalog(symbol);

CREATE INDEX IF NOT EXISTS idx_instrument_catalog_category 
    ON instrument_catalog(category);

COMMENT ON TABLE instrument_catalog IS 'Catalog of instruments across categories (built-in and custom).';
