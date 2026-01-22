"""
Cross-Check Analysis Module
Step 3.1: Compare manual Excel analysis with automated Alpha Analyst analysis
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz
import io

from tradingagents.database.config import get_supabase
from tradingagents.dataflows.validation_engine import ValidationEngine


class CrossCheckAnalyzer:
    """
    Cross-Check Analysis Engine
    Facilitates side-by-side comparison of manual Excel analysis vs automated Alpha Analyst analysis.
    """
    
    def __init__(self, timezone: str = 'Asia/Dubai'):
        """
        Initialize cross-check analyzer.
        
        Args:
            timezone: Timezone for all calculations (default: GMT+4)
        """
        self.tz = pytz.timezone(timezone)
        self.utc_tz = pytz.timezone('UTC')
        self.supabase = get_supabase()
        self.validator = ValidationEngine(timezone=timezone)
    
    def export_automated_results_to_excel(
        self,
        provider_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        signal_ids: Optional[List[int]] = None
    ) -> bytes:
        """
        Step 1: Export automated analysis results to Excel for manual review.
        
        Creates an Excel file with automated results that can be manually reviewed
        and compared in Excel.
        
        Args:
            provider_name: Optional provider name filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            signal_ids: Optional list of specific signal IDs
            
        Returns:
            Excel file as bytes (BytesIO)
        """
        if not self.supabase:
            raise ValueError("Database not available")
        
        try:
            # Fetch automated analysis results
            query = self.supabase.table('analysis_results').select('*')
            query = query.eq('analysis_method', 'automated')
            
            if provider_name:
                query = query.eq('provider_name', provider_name)
            if start_date:
                query = query.gte('signal_date', start_date.isoformat())
            if end_date:
                query = query.lte('signal_date', end_date.isoformat())
            
            result = query.order('signal_date', desc=True).execute()
            analysis_results = result.data if result.data else []
            
            if signal_ids:
                # Filter by signal IDs
                analysis_results = [r for r in analysis_results if r.get('signal_id') in signal_ids]
            
            if not analysis_results:
                raise ValueError("No automated analysis results found")
            
            # Fetch corresponding signal details
            signal_ids_list = [r.get('signal_id') for r in analysis_results if r.get('signal_id')]
            signals_query = self.supabase.table('signal_provider_signals').select('*')
            signals_query = signals_query.in_('id', signal_ids_list)
            signals_result = signals_query.execute()
            signals_dict = {s.get('id'): s for s in (signals_result.data if signals_result.data else [])}
            
            # Build Excel data
            excel_data = []
            for analysis in analysis_results:
                signal_id = analysis.get('signal_id')
                signal = signals_dict.get(signal_id, {})
                
                # Format timestamps for Excel (timezone-naive, readable format)
                def format_datetime(dt_str):
                    if not dt_str:
                        return None
                    try:
                        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                        if dt.tzinfo:
                            dt = dt.astimezone(self.tz).replace(tzinfo=None)
                        return dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        return str(dt_str) if dt_str else None
                
                row = {
                    'Signal ID': signal_id,
                    'Provider': analysis.get('provider_name', ''),
                    'Symbol': analysis.get('symbol', ''),
                    'Signal Date': format_datetime(analysis.get('signal_date')),
                    'Action': signal.get('action', '').upper() if signal else '',
                    'Entry Price': signal.get('entry_price'),
                    'Stop Loss': signal.get('stop_loss'),
                    'Target 1': signal.get('target_1'),
                    'Target 2': signal.get('target_2'),
                    'Target 3': signal.get('target_3'),
                    # Automated Results
                    'Auto TP1 Hit': 'Yes' if analysis.get('tp1_hit') else 'No',
                    'Auto TP1 Hit Date': format_datetime(analysis.get('tp1_hit_datetime')),
                    'Auto TP2 Hit': 'Yes' if analysis.get('tp2_hit') else 'No',
                    'Auto TP2 Hit Date': format_datetime(analysis.get('tp2_hit_datetime')),
                    'Auto TP3 Hit': 'Yes' if analysis.get('tp3_hit') else 'No',
                    'Auto TP3 Hit Date': format_datetime(analysis.get('tp3_hit_datetime')),
                    'Auto SL Hit': 'Yes' if analysis.get('sl_hit') else 'No',
                    'Auto SL Hit Date': format_datetime(analysis.get('sl_hit_datetime')),
                    'Auto Final Status': analysis.get('final_status', ''),
                    'Auto Max Profit': analysis.get('max_profit'),
                    'Auto Max Drawdown': analysis.get('max_drawdown'),
                    'Auto Hold Time (hours)': analysis.get('hold_time_hours'),
                    # Manual Review Columns (empty for user to fill)
                    'Manual TP1 Hit': '',
                    'Manual TP1 Hit Date': '',
                    'Manual TP2 Hit': '',
                    'Manual TP2 Hit Date': '',
                    'Manual TP3 Hit': '',
                    'Manual TP3 Hit Date': '',
                    'Manual SL Hit': '',
                    'Manual SL Hit Date': '',
                    'Manual Final Status': '',
                    'Notes': ''
                }
                excel_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(excel_data)
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Automated Results', index=False)
                
                # Add instructions sheet
                instructions = pd.DataFrame({
                    'Column': [
                        'Signal ID',
                        'Auto TP1 Hit',
                        'Manual TP1 Hit',
                        'Notes'
                    ],
                    'Description': [
                        'Unique signal identifier',
                        'Automated analysis result (Yes/No)',
                        'Fill this with your manual analysis (Yes/No)',
                        'Add any notes about discrepancies'
                    ]
                })
                instructions.to_excel(writer, sheet_name='Instructions', index=False)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            raise ValueError(f"Failed to export automated results: {str(e)}")
    
    def import_manual_analysis_from_excel(
        self,
        excel_file_path: str,
        provider_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Step 2: Import manual analysis from Excel file.
        
        Reads the Excel file that was exported (with manual analysis filled in)
        and performs validation.
        
        Args:
            excel_file_path: Path to Excel file with manual analysis
            provider_name: Optional provider name filter
            
        Returns:
            List of validation results
        """
        return self.validator.validate_from_excel(excel_file_path, provider_name)
    
    def generate_comparison_report(
        self,
        validation_results: List[Dict],
        include_details: bool = True
    ) -> pd.DataFrame:
        """
        Step 3: Generate side-by-side comparison report.
        
        Creates a detailed comparison report showing automated vs manual results
        side-by-side with discrepancy analysis.
        
        Args:
            validation_results: List of validation results from validate_from_excel
            include_details: Whether to include detailed discrepancy information
            
        Returns:
            DataFrame with comparison report
        """
        if not validation_results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for validation in validation_results:
            if 'error' in validation:
                continue
            
            signal_id = validation.get('signal_id')
            discrepancy_details = validation.get('discrepancy_details', {})
            
            # Fetch signal and automated result for context
            signal = self._get_signal(signal_id)
            automated_result = self._get_automated_result(signal_id)
            
            if not signal or not automated_result:
                continue
            
            row = {
                'Signal ID': signal_id,
                'Provider': automated_result.get('provider_name', ''),
                'Symbol': automated_result.get('symbol', ''),
                'Signal Date': self._format_datetime(automated_result.get('signal_date')),
                'Action': signal.get('action', '').upper(),
                'Entry Price': signal.get('entry_price'),
                'Stop Loss': signal.get('stop_loss'),
                'Target 1': signal.get('target_1'),
                'Target 2': signal.get('target_2'),
                'Target 3': signal.get('target_3'),
                # TP1 Comparison
                'Auto TP1': 'Yes' if automated_result.get('tp1_hit') else 'No',
                'Auto TP1 Date': self._format_datetime(automated_result.get('tp1_hit_datetime')),
                'Manual TP1': 'Yes' if discrepancy_details.get('tp1', {}).get('manual') else 'No',
                'Manual TP1 Date': self._format_datetime(discrepancy_details.get('tp1', {}).get('manual_datetime')),
                'TP1 Match': '✓' if validation.get('tp1_match') else '✗',
                'TP1 Diff (min)': validation.get('tp1_timestamp_diff_minutes'),
                # TP2 Comparison
                'Auto TP2': 'Yes' if automated_result.get('tp2_hit') else 'No',
                'Auto TP2 Date': self._format_datetime(automated_result.get('tp2_hit_datetime')),
                'Manual TP2': 'Yes' if discrepancy_details.get('tp2', {}).get('manual') else 'No',
                'Manual TP2 Date': self._format_datetime(discrepancy_details.get('tp2', {}).get('manual_datetime')),
                'TP2 Match': '✓' if validation.get('tp2_match') else '✗',
                'TP2 Diff (min)': validation.get('tp2_timestamp_diff_minutes'),
                # TP3 Comparison
                'Auto TP3': 'Yes' if automated_result.get('tp3_hit') else 'No',
                'Auto TP3 Date': self._format_datetime(automated_result.get('tp3_hit_datetime')),
                'Manual TP3': 'Yes' if discrepancy_details.get('tp3', {}).get('manual') else 'No',
                'Manual TP3 Date': self._format_datetime(discrepancy_details.get('tp3', {}).get('manual_datetime')),
                'TP3 Match': '✓' if validation.get('tp3_match') else '✗',
                'TP3 Diff (min)': validation.get('tp3_timestamp_diff_minutes'),
                # SL Comparison
                'Auto SL': 'Yes' if automated_result.get('sl_hit') else 'No',
                'Auto SL Date': self._format_datetime(automated_result.get('sl_hit_datetime')),
                'Manual SL': 'Yes' if discrepancy_details.get('sl', {}).get('manual') else 'No',
                'Manual SL Date': self._format_datetime(discrepancy_details.get('sl', {}).get('manual_datetime')),
                'SL Match': '✓' if validation.get('sl_match') else '✗',
                'SL Diff (min)': validation.get('sl_timestamp_diff_minutes'),
                # Status Comparison
                'Auto Status': automated_result.get('final_status', ''),
                'Manual Status': discrepancy_details.get('status', {}).get('manual', ''),
                'Status Match': '✓' if validation.get('status_match') else '✗',
                # Discrepancy Info
                'Discrepancy Type': validation.get('discrepancy_type', ''),
                'Severity': validation.get('discrepancy_severity', ''),
                'Validation Date': self._format_datetime(validation.get('validation_date'))
            }
            
            if include_details:
                row['Max Profit'] = automated_result.get('max_profit')
                row['Max Drawdown'] = automated_result.get('max_drawdown')
                row['Hold Time (hrs)'] = automated_result.get('hold_time_hours')
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def export_comparison_report_to_excel(
        self,
        comparison_df: pd.DataFrame,
        filename_prefix: str = "cross_check_report"
    ) -> bytes:
        """
        Export comparison report to Excel file.
        
        Args:
            comparison_df: DataFrame from generate_comparison_report
            filename_prefix: Prefix for the filename
            
        Returns:
            Excel file as bytes
        """
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main comparison sheet
            comparison_df.to_excel(writer, sheet_name='Comparison Report', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Signals Validated',
                    'Perfect Matches',
                    'Discrepancies Found',
                    'Match Rate (%)',
                    'Critical Issues',
                    'Warnings',
                    'Minor Issues'
                ],
                'Value': [
                    len(comparison_df),
                    len(comparison_df[comparison_df['Discrepancy Type'] == 'NO_MISMATCH']),
                    len(comparison_df[comparison_df['Discrepancy Type'] != 'NO_MISMATCH']),
                    round(len(comparison_df[comparison_df['Discrepancy Type'] == 'NO_MISMATCH']) / len(comparison_df) * 100, 2) if len(comparison_df) > 0 else 0,
                    len(comparison_df[comparison_df['Severity'] == 'CRITICAL']),
                    len(comparison_df[comparison_df['Severity'] == 'WARNING']),
                    len(comparison_df[comparison_df['Severity'] == 'MINOR'])
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    def _get_signal(self, signal_id: int) -> Optional[Dict]:
        """Fetch signal from database."""
        if not self.supabase:
            return None
        
        try:
            result = self.supabase.table('signal_provider_signals').select('*').eq('id', signal_id).execute()
            return result.data[0] if result.data else None
        except:
            return None
    
    def _get_automated_result(self, signal_id: int) -> Optional[Dict]:
        """Fetch automated analysis result from database."""
        return self.validator._get_automated_result(signal_id)
    
    def _format_datetime(self, dt_str: Optional[str]) -> Optional[str]:
        """Format datetime string for display."""
        if not dt_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            if dt.tzinfo:
                dt = dt.astimezone(self.tz).replace(tzinfo=None)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return str(dt_str) if dt_str else None

