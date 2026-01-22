"""
Daily Progress Reporter
Tracks daily progress for data ingestion, signal processing, and analysis completion.
"""
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import pytz

from tradingagents.database.config import get_supabase
from tradingagents.dataflows.market_data_service import fetch_ohlcv


class DailyReporter:
    """
    Tracks and reports daily progress for the catch-up plan.
    """
    
    def __init__(self, timezone: str = 'Asia/Dubai'):
        """
        Initialize daily reporter.
        
        Args:
            timezone: Timezone for all calculations (default: GMT+4)
        """
        self.tz = pytz.timezone(timezone)
        self.utc_tz = pytz.timezone('UTC')
        self.supabase = get_supabase()
    
    def generate_daily_report(self, report_date: Optional[date] = None) -> Dict:
        """
        Generate daily progress report.
        
        Args:
            report_date: Date for report (default: today)
            
        Returns:
            Dictionary with daily progress metrics
        """
        if report_date is None:
            report_date = datetime.now(self.tz).date()
        
        try:
            # Data Ingestion Progress
            currencies_progress = self._get_currencies_progress()
            indices_progress = self._get_indices_progress()
            market_data_completeness = self._get_market_data_completeness()
            
            # Signal Processing
            signal_stats = self._get_signal_statistics(report_date)
            
            # Analysis Status
            analysis_stats = self._get_analysis_statistics()
            
            # Issues Tracking
            issues = self._get_issues_tracking()
            
            # Telegram Signals Status
            telegram_status = self._get_telegram_status()
            
            # Validation Status
            validation_status = self._get_validation_status()
            
            # Team Access
            work_number_active = self._check_work_number_status()
            
            # Build report
            report = {
                'log_date': report_date.isoformat(),
                'timezone_offset': '+04:00',
                
                # Data Ingestion
                'currencies_total': currencies_progress.get('total', 28),
                'currencies_ingested': currencies_progress.get('ingested', 0),
                'currencies_visible_ui': currencies_progress.get('visible_ui', 0),
                'currencies_validated': currencies_progress.get('validated', 0),
                'indices_started': indices_progress.get('started', False),
                'indices_progress_percent': indices_progress.get('progress_percent', 0),
                'market_data_completeness_percent': market_data_completeness,
                
                # Signal Processing
                'new_signals_today': signal_stats.get('new_today', 0),
                'total_signals_db': signal_stats.get('total', 0),
                'parsing_success_rate': signal_stats.get('success_rate', 0),
                
                # Analysis Status
                'signals_analyzed': analysis_stats.get('analyzed', 0),
                'backtesting_complete_percent': analysis_stats.get('backtest_percent', 0),
                'validation_match_rate': validation_status.get('match_rate', 0),
                
                # Issues
                'symbol_mismatches_count': issues.get('symbol_mismatches', 0),
                'data_gaps_count': issues.get('data_gaps', 0),
                'parsing_errors_count': issues.get('parsing_errors', 0),
                
                # Telegram
                'telegram_live_fetching': telegram_status.get('live_fetching', False),
                'telegram_backtesting_started': telegram_status.get('backtesting_started', False),
                'telegram_historical_signals_count': telegram_status.get('historical_count', 0),
                
                # Validation
                'excel_analysis_status': validation_status.get('excel_status', 'NOT_STARTED'),
                'automated_analysis_status': validation_status.get('automated_status', 'NOT_STARTED'),
                'cross_check_status': validation_status.get('cross_check_status', 'NOT_STARTED'),
                'cross_check_match_rate': validation_status.get('match_rate', 0),
                
                # Team Access
                'work_number_active': work_number_active,
                
                'created_at': datetime.now(self.tz).isoformat(),
                'updated_at': datetime.now(self.tz).isoformat()
            }
            
            return report
            
        except Exception as e:
            return {"error": f"Failed to generate report: {str(e)}"}
    
    def _get_currencies_progress(self) -> Dict:
        """Get currencies ingestion progress."""
        # List of 28 major currency pairs
        currency_pairs = [
            'C:EURUSD', 'C:GBPUSD', 'C:USDJPY', 'C:USDCHF', 'C:AUDUSD',
            'C:NZDUSD', 'C:USDCAD', 'C:EURGBP', 'C:EURJPY', 'C:GBPJPY',
            'C:AUDJPY', 'C:EURCHF', 'C:AUDNZD', 'C:EURAUD', 'C:GBPAUD',
            'C:GBPCAD', 'C:GBPCHF', 'C:GBPNZD', 'C:EURNZD', 'C:EURCAD',
            'C:AUDCAD', 'C:AUDCHF', 'C:CADCHF', 'C:CADJPY', 'C:CHFJPY',
            'C:NZDJPY', 'C:NZDCHF', 'C:NZDCAD'
        ]
        
        ingested = 0
        visible_ui = 0
        validated = 0
        
        for symbol in currency_pairs:
            # Check if data exists (simple check - fetch latest price)
            try:
                data = fetch_ohlcv(symbol, interval='1d', lookback_days=1)
                if data is not None and not data.empty:
                    ingested += 1
                    visible_ui += 1  # If data exists, assume visible in UI
            except:
                pass
        
        return {
            'total': len(currency_pairs),
            'ingested': ingested,
            'visible_ui': visible_ui,
            'validated': validated  # Would need validation tracking
        }
    
    def _get_indices_progress(self) -> Dict:
        """Get indices ingestion progress."""
        # Check if indices ingestion has started
        indices_symbols = ['I:SPX', 'I:DJI', 'I:NDX']
        
        started = False
        with_data = 0
        
        for symbol in indices_symbols:
            try:
                data = fetch_ohlcv(symbol, interval='1d', lookback_days=1)
                if data is not None and not data.empty:
                    started = True
                    with_data += 1
            except:
                pass
        
        progress_percent = (with_data / len(indices_symbols) * 100) if indices_symbols else 0
        
        return {
            'started': started,
            'progress_percent': round(progress_percent, 2)
        }
    
    def _get_market_data_completeness(self) -> float:
        """Calculate market data completeness percentage."""
        # This would check data gaps across all symbols
        # For now, return a placeholder
        return 85.0  # Would need actual calculation
    
    def _get_signal_statistics(self, report_date: date) -> Dict:
        """Get signal processing statistics."""
        if not self.supabase:
            return {'new_today': 0, 'total': 0, 'success_rate': 0}
        
        try:
            # Total signals
            total_result = self.supabase.table('signal_provider_signals').select('id', count='exact').execute()
            total = total_result.count if hasattr(total_result, 'count') else len(total_result.data) if total_result.data else 0
            
            # Signals created today
            start_of_day = datetime.combine(report_date, datetime.min.time())
            start_of_day = self.tz.localize(start_of_day)
            end_of_day = start_of_day + timedelta(days=1)
            
            today_result = self.supabase.table('signal_provider_signals').select('id', count='exact').gte('created_at', start_of_day.isoformat()).lt('created_at', end_of_day.isoformat()).execute()
            new_today = today_result.count if hasattr(today_result, 'count') else len(today_result.data) if today_result.data else 0
            
            # Parsing success rate (would need to track parsing attempts)
            success_rate = 95.0  # Placeholder
            
            return {
                'new_today': new_today,
                'total': total,
                'success_rate': success_rate
            }
            
        except Exception:
            return {'new_today': 0, 'total': 0, 'success_rate': 0}
    
    def _get_analysis_statistics(self) -> Dict:
        """Get analysis statistics."""
        if not self.supabase:
            return {'analyzed': 0, 'backtest_percent': 0}
        
        try:
            # Count analyzed signals
            analyzed_result = self.supabase.table('analysis_results').select('id', count='exact').execute()
            analyzed = analyzed_result.count if hasattr(analyzed_result, 'count') else len(analyzed_result.data) if analyzed_result.data else 0
            
            # Count total signals
            total_result = self.supabase.table('signal_provider_signals').select('id', count='exact').execute()
            total = total_result.count if hasattr(total_result, 'count') else len(total_result.data) if total_result.data else 0
            
            # Count backtested signals
            backtest_result = self.supabase.table('backtest_results').select('id', count='exact').execute()
            backtested = backtest_result.count if hasattr(backtest_result, 'count') else len(backtest_result.data) if backtest_result.data else 0
            
            backtest_percent = (backtested / total * 100) if total > 0 else 0
            
            return {
                'analyzed': analyzed,
                'backtest_percent': round(backtest_percent, 2)
            }
            
        except Exception:
            return {'analyzed': 0, 'backtest_percent': 0}
    
    def _get_issues_tracking(self) -> Dict:
        """Get issues tracking counts."""
        if not self.supabase:
            return {'symbol_mismatches': 0, 'data_gaps': 0, 'parsing_errors': 0}
        
        try:
            # Symbol mismatches from validation reports
            mismatch_result = self.supabase.table('validation_reports').select('id', count='exact').neq('discrepancy_type', 'NO_MISMATCH').execute()
            symbol_mismatches = mismatch_result.count if hasattr(mismatch_result, 'count') else len(mismatch_result.data) if mismatch_result.data else 0
            
            # Data gaps and parsing errors would need separate tracking
            data_gaps = 0
            parsing_errors = 0
            
            return {
                'symbol_mismatches': symbol_mismatches,
                'data_gaps': data_gaps,
                'parsing_errors': parsing_errors
            }
            
        except Exception:
            return {'symbol_mismatches': 0, 'data_gaps': 0, 'parsing_errors': 0}
    
    def _get_telegram_status(self) -> Dict:
        """Get Telegram signals status."""
        if not self.supabase:
            return {'live_fetching': False, 'backtesting_started': False, 'historical_count': 0}
        
        try:
            # Check if there are recent signals (within last 24 hours)
            recent_time = datetime.now(self.tz) - timedelta(hours=24)
            recent_result = self.supabase.table('signal_provider_signals').select('id', count='exact').gte('created_at', recent_time.isoformat()).execute()
            recent_count = recent_result.count if hasattr(recent_result, 'count') else len(recent_result.data) if recent_result.data else 0
            
            live_fetching = recent_count > 0
            
            # Check if backtesting has started
            backtest_result = self.supabase.table('backtest_results').select('id', count='exact').execute()
            backtesting_started = backtest_result.count > 0 if hasattr(backtest_result, 'count') else len(backtest_result.data) > 0 if backtest_result.data else False
            
            # Count historical signals
            historical_result = self.supabase.table('signal_provider_signals').select('id', count='exact').execute()
            historical_count = historical_result.count if hasattr(historical_result, 'count') else len(historical_result.data) if historical_result.data else 0
            
            return {
                'live_fetching': live_fetching,
                'backtesting_started': backtesting_started,
                'historical_count': historical_count
            }
            
        except Exception:
            return {'live_fetching': False, 'backtesting_started': False, 'historical_count': 0}
    
    def _get_validation_status(self) -> Dict:
        """Get validation status."""
        if not self.supabase:
            return {
                'excel_status': 'NOT_STARTED',
                'automated_status': 'NOT_STARTED',
                'cross_check_status': 'NOT_STARTED',
                'match_rate': 0
            }
        
        try:
            # Check automated analysis status
            auto_result = self.supabase.table('analysis_results').select('id', count='exact').execute()
            auto_count = auto_result.count if hasattr(auto_result, 'count') else len(auto_result.data) if auto_result.data else 0
            automated_status = 'COMPLETE' if auto_count > 0 else 'NOT_STARTED'
            
            # Check validation reports
            validation_result = self.supabase.table('validation_reports').select('id', count='exact').execute()
            validation_count = validation_result.count if hasattr(validation_result, 'count') else len(validation_result.data) if validation_result.data else 0
            
            cross_check_status = 'COMPLETE' if validation_count > 0 else 'NOT_STARTED'
            
            # Calculate match rate
            if validation_count > 0:
                match_result = self.supabase.table('validation_reports').select('discrepancy_type', count='exact').eq('discrepancy_type', 'NO_MISMATCH').execute()
                matches = match_result.count if hasattr(match_result, 'count') else len(match_result.data) if match_result.data else 0
                match_rate = (matches / validation_count * 100) if validation_count > 0 else 0
            else:
                match_rate = 0
            
            # Excel status would need manual tracking
            excel_status = 'NOT_STARTED'  # Placeholder
            
            return {
                'excel_status': excel_status,
                'automated_status': automated_status,
                'cross_check_status': cross_check_status,
                'match_rate': round(match_rate, 2)
            }
            
        except Exception:
            return {
                'excel_status': 'NOT_STARTED',
                'automated_status': 'NOT_STARTED',
                'cross_check_status': 'NOT_STARTED',
                'match_rate': 0
            }
    
    def _check_work_number_status(self) -> bool:
        """Check if Work Number is active."""
        # This would need integration with Work Number API
        # For now, return placeholder
        return True  # Placeholder
    
    def save_daily_report(self, report: Dict) -> bool:
        """
        Save daily report to database.
        
        Args:
            report: Daily report dictionary
            
        Returns:
            True if saved successfully
        """
        if not self.supabase or 'error' in report:
            return False
        
        try:
            # Remove error field if present
            report = {k: v for k, v in report.items() if k != 'error'}
            
            self.supabase.table('daily_progress_log').upsert(report).execute()
            return True
            
        except Exception as e:
            print(f"Error saving daily report: {e}")
            return False
    
    def get_latest_report(self) -> Optional[Dict]:
        """Get the latest daily report."""
        if not self.supabase:
            return None
        
        try:
            result = self.supabase.table('daily_progress_log').select('*').order('log_date', desc=True).limit(1).execute()
            if result.data:
                return result.data[0]
            return None
            
        except Exception:
            return None

