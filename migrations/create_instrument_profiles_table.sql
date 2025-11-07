CREATE TABLE instrument_profiles (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  symbol text NOT NULL,
  profile_text text,
  embedding_vector jsonb,
  profile_data jsonb,
  generated_at timestamp with time zone DEFAULT now(),
  analysis_timestamp timestamp with time zone,
  source text DEFAULT 'instrument_profile'
);

-- Optional indexes for faster lookups
CREATE INDEX idx_instrument_profiles_symbol ON instrument_profiles(symbol);
CREATE INDEX idx_instrument_profiles_generated_at ON instrument_profiles(generated_at DESC);

