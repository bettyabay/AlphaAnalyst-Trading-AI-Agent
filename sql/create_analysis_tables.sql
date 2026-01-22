-- Analysis Results Table: Stores automated analysis outputs
CREATE TABLE IF NOT EXISTS analysis_results (
    id BIGSERIAL PRIMARY KEY,
    signal_id BIGINT REFERENCES signal_provider_signals(id) ON DELETE CASCADE,
    provider_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    signal_date TIMESTAMPTZ NOT NULL,
    
    -- Analysis Results
    tp1_hit BOOLEAN DEFAULT FALSE,
    tp1_hit_datetime TIMESTAMPTZ,
    tp2_hit BOOLEAN DEFAULT FALSE,
    tp2_hit_datetime TIMESTAMPTZ,
    tp3_hit BOOLEAN DEFAULT FALSE,
    tp3_hit_datetime TIMESTAMPTZ,
    sl_hit BOOLEAN DEFAULT FALSE,
    sl_hit_datetime TIMESTAMPTZ,
    
    -- Performance Metrics
    max_profit NUMERIC(18, 6),
    max_drawdown NUMERIC(18, 6),
    final_status VARCHAR(20), -- 'TP1', 'TP2', 'TP3', 'SL', 'OPEN', 'EXPIRED'
    hold_time_hours NUMERIC(10, 2),
    
    -- Analysis Metadata
    analysis_date TIMESTAMPTZ DEFAULT NOW(),
    analysis_method VARCHAR(50) DEFAULT 'automated',
    timezone_offset VARCHAR(10) DEFAULT '+04:00',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(signal_id, analysis_method)
);

CREATE INDEX IF NOT EXISTS idx_analysis_results_signal_id ON analysis_results(signal_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_symbol ON analysis_results(symbol);
CREATE INDEX IF NOT EXISTS idx_analysis_results_provider ON analysis_results(provider_name);
CREATE INDEX IF NOT EXISTS idx_analysis_results_date ON analysis_results(signal_date);

-- Validation Reports Table: Stores comparison results between manual and automated
CREATE TABLE IF NOT EXISTS validation_reports (
    id BIGSERIAL PRIMARY KEY,
    signal_id BIGINT REFERENCES signal_provider_signals(id) ON DELETE CASCADE,
    analysis_result_id BIGINT REFERENCES analysis_results(id) ON DELETE CASCADE,
    
    -- Comparison Results
    tp1_match BOOLEAN,
    tp1_timestamp_diff_minutes INTEGER,
    tp2_match BOOLEAN,
    tp2_timestamp_diff_minutes INTEGER,
    tp3_match BOOLEAN,
    tp3_timestamp_diff_minutes INTEGER,
    sl_match BOOLEAN,
    sl_timestamp_diff_minutes INTEGER,
    status_match BOOLEAN,
    
    -- Discrepancy Details
    discrepancy_type VARCHAR(50), -- 'TP_MISMATCH', 'SL_MISMATCH', 'TIMESTAMP_MISMATCH', 'NO_MISMATCH'
    discrepancy_severity VARCHAR(20), -- 'CRITICAL', 'WARNING', 'MINOR'
    discrepancy_details JSONB,
    
    -- Manual Analysis Source
    manual_analysis_source VARCHAR(100), -- 'EXCEL', 'MANUAL_ENTRY'
    manual_analysis_date TIMESTAMPTZ,
    
    -- Validation Metadata
    validation_date TIMESTAMPTZ DEFAULT NOW(),
    validator_name VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validation_reports_signal_id ON validation_reports(signal_id);
CREATE INDEX IF NOT EXISTS idx_validation_reports_discrepancy ON validation_reports(discrepancy_type);
CREATE INDEX IF NOT EXISTS idx_validation_reports_severity ON validation_reports(discrepancy_severity);

-- Backtest Results Table: Stores backtesting simulation results
CREATE TABLE IF NOT EXISTS backtest_results (
    id BIGSERIAL PRIMARY KEY,
    signal_id BIGINT REFERENCES signal_provider_signals(id) ON DELETE CASCADE,
    provider_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    
    -- Backtest Parameters
    backtest_start_date TIMESTAMPTZ NOT NULL,
    backtest_end_date TIMESTAMPTZ,
    initial_capital NUMERIC(18, 2),
    position_size_percent NUMERIC(5, 2), -- Percentage of capital per trade
    
    -- Trade Simulation Results
    entry_datetime TIMESTAMPTZ,
    exit_datetime TIMESTAMPTZ,
    entry_price NUMERIC(18, 6),
    exit_price NUMERIC(18, 6),
    position_size NUMERIC(18, 2),
    
    -- P&L Calculation
    profit_loss NUMERIC(18, 2),
    profit_loss_percent NUMERIC(10, 4),
    commission NUMERIC(18, 2) DEFAULT 0,
    net_profit_loss NUMERIC(18, 2),
    
    -- Exit Details
    exit_reason VARCHAR(50), -- 'TP1', 'TP2', 'TP3', 'SL', 'MANUAL', 'EXPIRED'
    tp1_exit_percent NUMERIC(5, 2) DEFAULT 33.33,
    tp2_exit_percent NUMERIC(5, 2) DEFAULT 33.33,
    tp3_exit_percent NUMERIC(5, 2) DEFAULT 33.34,
    
    -- Performance Metrics
    max_profit NUMERIC(18, 2),
    max_drawdown NUMERIC(18, 2),
    hold_time_hours NUMERIC(10, 2),
    
    -- Backtest Metadata
    backtest_date TIMESTAMPTZ DEFAULT NOW(),
    backtest_method VARCHAR(50) DEFAULT 'historical_simulation',
    timezone_offset VARCHAR(10) DEFAULT '+04:00',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_results_signal_id ON backtest_results(signal_id);
CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol ON backtest_results(symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_results_provider ON backtest_results(provider_name);
CREATE INDEX IF NOT EXISTS idx_backtest_results_date ON backtest_results(backtest_start_date);

-- Daily Progress Log Table: Tracks daily progress for catch-up
CREATE TABLE IF NOT EXISTS daily_progress_log (
    id BIGSERIAL PRIMARY KEY,
    log_date DATE NOT NULL,
    timezone_offset VARCHAR(10) DEFAULT '+04:00',
    
    -- Data Ingestion Progress
    currencies_total INTEGER DEFAULT 28,
    currencies_ingested INTEGER DEFAULT 0,
    currencies_visible_ui INTEGER DEFAULT 0,
    currencies_validated INTEGER DEFAULT 0,
    indices_started BOOLEAN DEFAULT FALSE,
    indices_progress_percent NUMERIC(5, 2) DEFAULT 0,
    market_data_completeness_percent NUMERIC(5, 2) DEFAULT 0,
    
    -- Signal Processing
    new_signals_today INTEGER DEFAULT 0,
    total_signals_db INTEGER DEFAULT 0,
    parsing_success_rate NUMERIC(5, 2) DEFAULT 0,
    
    -- Analysis Status
    signals_analyzed INTEGER DEFAULT 0,
    backtesting_complete_percent NUMERIC(5, 2) DEFAULT 0,
    validation_match_rate NUMERIC(5, 2) DEFAULT 0,
    
    -- Issues Tracking
    symbol_mismatches_count INTEGER DEFAULT 0,
    data_gaps_count INTEGER DEFAULT 0,
    parsing_errors_count INTEGER DEFAULT 0,
    
    -- Telegram Signals Status
    telegram_live_fetching BOOLEAN DEFAULT FALSE,
    telegram_backtesting_started BOOLEAN DEFAULT FALSE,
    telegram_historical_signals_count INTEGER DEFAULT 0,
    
    -- Validation Status
    excel_analysis_status VARCHAR(20) DEFAULT 'NOT_STARTED', -- 'NOT_STARTED', 'IN_PROGRESS', 'COMPLETE'
    automated_analysis_status VARCHAR(20) DEFAULT 'NOT_STARTED',
    cross_check_status VARCHAR(20) DEFAULT 'NOT_STARTED',
    cross_check_match_rate NUMERIC(5, 2),
    
    -- Team Access
    work_number_active BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(log_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_progress_log_date ON daily_progress_log(log_date);

-- Provider Performance Summary Table: Aggregated metrics per provider
CREATE TABLE IF NOT EXISTS provider_performance_summary (
    id BIGSERIAL PRIMARY KEY,
    provider_name VARCHAR(100) NOT NULL,
    calculation_date DATE NOT NULL,
    
    -- Signal Counts
    total_signals INTEGER DEFAULT 0,
    analyzed_signals INTEGER DEFAULT 0,
    
    -- Success Rates
    tp1_success_count INTEGER DEFAULT 0,
    tp1_success_rate NUMERIC(5, 2) DEFAULT 0,
    tp2_success_count INTEGER DEFAULT 0,
    tp2_success_rate NUMERIC(5, 2) DEFAULT 0,
    tp3_success_count INTEGER DEFAULT 0,
    tp3_success_rate NUMERIC(5, 2) DEFAULT 0,
    sl_hit_count INTEGER DEFAULT 0,
    sl_hit_rate NUMERIC(5, 2) DEFAULT 0,
    
    -- Overall Metrics
    win_rate NUMERIC(5, 2) DEFAULT 0, -- (TP1 + TP2 + TP3) / Total
    risk_reward_ratio NUMERIC(10, 4) DEFAULT 0,
    average_hold_time_hours NUMERIC(10, 2) DEFAULT 0,
    
    -- Backtest Metrics
    total_pnl NUMERIC(18, 2) DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    average_win NUMERIC(18, 2) DEFAULT 0,
    average_loss NUMERIC(18, 2) DEFAULT 0,
    sharpe_ratio NUMERIC(10, 4),
    
    -- Metadata
    timezone_offset VARCHAR(10) DEFAULT '+04:00',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(provider_name, calculation_date)
);

CREATE INDEX IF NOT EXISTS idx_provider_performance_provider ON provider_performance_summary(provider_name);
CREATE INDEX IF NOT EXISTS idx_provider_performance_date ON provider_performance_summary(calculation_date);
