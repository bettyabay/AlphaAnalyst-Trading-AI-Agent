-- Add MFE/MAE columns to backtest_results table
-- Trade Efficiency Analysis Module - Database Migration

-- Add efficiency columns
ALTER TABLE backtest_results 
ADD COLUMN IF NOT EXISTS mfe NUMERIC(18, 6),
ADD COLUMN IF NOT EXISTS mae NUMERIC(18, 6),
ADD COLUMN IF NOT EXISTS mfe_pips NUMERIC(10, 2),
ADD COLUMN IF NOT EXISTS mae_pips NUMERIC(10, 2),
ADD COLUMN IF NOT EXISTS mae_r NUMERIC(10, 4),
ADD COLUMN IF NOT EXISTS mfe_r NUMERIC(10, 4),
ADD COLUMN IF NOT EXISTS efficiency_confidence VARCHAR(10) DEFAULT 'HIGH';

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_backtest_results_efficiency 
ON backtest_results(mae_pips, mfe_pips);

-- Add comment to table
COMMENT ON COLUMN backtest_results.mfe IS 'Maximum Favorable Excursion (price units)';
COMMENT ON COLUMN backtest_results.mae IS 'Maximum Adverse Excursion (price units)';
COMMENT ON COLUMN backtest_results.mfe_pips IS 'Maximum Favorable Excursion in pips';
COMMENT ON COLUMN backtest_results.mae_pips IS 'Maximum Adverse Excursion in pips';
COMMENT ON COLUMN backtest_results.mae_r IS 'MAE normalized as R-multiple (MAE / Stop Loss Distance)';
COMMENT ON COLUMN backtest_results.mfe_r IS 'MFE normalized as R-multiple (MFE / Stop Loss Distance)';
COMMENT ON COLUMN backtest_results.efficiency_confidence IS 'Confidence level: HIGH (multiple bars) or LOW (same bar trade)';

