-- Create market_data_commodities_1min table
CREATE TABLE IF NOT EXISTS public.market_data_commodities_1min (
    symbol text NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    volume numeric,
    open_interest numeric,
    created_at timestamp with time zone DEFAULT now(),
    PRIMARY KEY (symbol, "timestamp")
);

-- Create market_data_indices_1min table
CREATE TABLE IF NOT EXISTS public.market_data_indices_1min (
    symbol text NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    volume numeric,
    open_interest numeric,
    created_at timestamp with time zone DEFAULT now(),
    PRIMARY KEY (symbol, "timestamp")
);

-- Create market_data_currencies_1min table
CREATE TABLE IF NOT EXISTS public.market_data_currencies_1min (
    symbol text NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    volume numeric,
    open_interest numeric,
    created_at timestamp with time zone DEFAULT now(),
    PRIMARY KEY (symbol, "timestamp")
);

-- Create market_data_stocks_1min table (Stocks)
CREATE TABLE IF NOT EXISTS public.market_data_stocks_1min (
    symbol text NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    volume numeric,
    created_at timestamp with time zone DEFAULT now(),
    PRIMARY KEY (symbol, "timestamp")
);

-- Enable Row Level Security (RLS) if needed
ALTER TABLE public.market_data_commodities_1min ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_data_indices_1min ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_data_currencies_1min ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_data_stocks_1min ENABLE ROW LEVEL SECURITY;

-- Create policies (optional, allows public access for development)
CREATE POLICY "Allow public read access" ON public.market_data_commodities_1min FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.market_data_commodities_1min FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access" ON public.market_data_commodities_1min FOR UPDATE USING (true);

CREATE POLICY "Allow public read access" ON public.market_data_indices_1min FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.market_data_indices_1min FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access" ON public.market_data_indices_1min FOR UPDATE USING (true);

CREATE POLICY "Allow public read access" ON public.market_data_currencies_1min FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.market_data_currencies_1min FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access" ON public.market_data_currencies_1min FOR UPDATE USING (true);

CREATE POLICY "Allow public read access" ON public.market_data_stocks_1min FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.market_data_stocks_1min FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access" ON public.market_data_stocks_1min FOR UPDATE USING (true);
