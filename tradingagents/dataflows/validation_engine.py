"""
Validation Engine
Compares automated analysis results with manual Excel analysis to ensure accuracy.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

from tradingagents.database.config import get_supabase


class ValidationEngine:
    """
    Validates automated analysis against manual analysis.
    Detects discrepancies and generates validation reports.
    """
    
    def __init__(self, timezone: str = 'Asia/Dubai', timestamp_tolerance_minutes: int = 5):
        """
        Initialize validation engine.
        
        Args:
            timezone: Timezone for all calculations (default: GMT+4)
            timestamp_tolerance_minutes: Tolerance for timestamp comparison (default: 5 minutes)
        """
        self.tz = pytz.timezone(timezone)
        self.utc_tz = pytz.timezone('UTC')
        self.timestamp_tolerance = timedelta(minutes=timestamp_tolerance_minutes)
        self.supabase = get_supabase()
    
    def validate_signal(
        self,
        signal_id: int,
        automated_result: Dict,
        manual_result: Dict
    ) -> Dict:
        """
        Validate a single signal by comparing automated vs manual results.
        
        Args:
            signal_id: Signal ID
            automated_result: Automated analysis result from analysis_results table
            manual_result: Manual analysis result (from Excel or manual entry)
                Expected fields: tp1_hit, tp1_hit_datetime, tp2_hit, tp2_hit_datetime,
                                tp3_hit, tp3_hit_datetime, sl_hit, sl_hit_datetime, final_status
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Compare TP1
            tp1_match, tp1_diff = self._compare_hit(
                automated_result.get('tp1_hit', False),
                automated_result.get('tp1_hit_datetime'),
                manual_result.get('tp1_hit', False),
                manual_result.get('tp1_hit_datetime')
            )
            
            # Compare TP2
            tp2_match, tp2_diff = self._compare_hit(
                automated_result.get('tp2_hit', False),
                automated_result.get('tp2_hit_datetime'),
                manual_result.get('tp2_hit', False),
                manual_result.get('tp2_hit_datetime')
            )
            
            # Compare TP3
            tp3_match, tp3_diff = self._compare_hit(
                automated_result.get('tp3_hit', False),
                automated_result.get('tp3_hit_datetime'),
                manual_result.get('tp3_hit', False),
                manual_result.get('tp3_hit_datetime')
            )
            
            # Compare SL
            sl_match, sl_diff = self._compare_hit(
                automated_result.get('sl_hit', False),
                automated_result.get('sl_hit_datetime'),
                manual_result.get('sl_hit', False),
                manual_result.get('sl_hit_datetime')
            )
            
            # Compare final status
            auto_status = automated_result.get('final_status', '').upper()
            manual_status = str(manual_result.get('final_status', '')).upper()
            status_match = (auto_status == manual_status)
            
            # Determine discrepancy type and severity
            discrepancy_type, discrepancy_severity = self._determine_discrepancy(
                tp1_match, tp2_match, tp3_match, sl_match, status_match
            )
            
            # Build discrepancy details
            discrepancy_details = {
                'tp1': {
                    'match': tp1_match,
                    'automated': automated_result.get('tp1_hit', False),
                    'automated_datetime': automated_result.get('tp1_hit_datetime'),
                    'manual': manual_result.get('tp1_hit', False),
                    'manual_datetime': manual_result.get('tp1_hit_datetime'),
                    'timestamp_diff_minutes': tp1_diff
                },
                'tp2': {
                    'match': tp2_match,
                    'automated': automated_result.get('tp2_hit', False),
                    'automated_datetime': automated_result.get('tp2_hit_datetime'),
                    'manual': manual_result.get('tp2_hit', False),
                    'manual_datetime': manual_result.get('tp2_hit_datetime'),
                    'timestamp_diff_minutes': tp2_diff
                },
                'tp3': {
                    'match': tp3_match,
                    'automated': automated_result.get('tp3_hit', False),
                    'automated_datetime': automated_result.get('tp3_hit_datetime'),
                    'manual': manual_result.get('tp3_hit', False),
                    'manual_datetime': manual_result.get('tp3_hit_datetime'),
                    'timestamp_diff_minutes': tp3_diff
                },
                'sl': {
                    'match': sl_match,
                    'automated': automated_result.get('sl_hit', False),
                    'automated_datetime': automated_result.get('sl_hit_datetime'),
                    'manual': manual_result.get('sl_hit', False),
                    'manual_datetime': manual_result.get('sl_hit_datetime'),
                    'timestamp_diff_minutes': sl_diff
                },
                'status': {
                    'match': status_match,
                    'automated': auto_status,
                    'manual': manual_status
                }
            }
            
            return {
                'signal_id': signal_id,
                'analysis_result_id': automated_result.get('id'),
                'tp1_match': tp1_match,
                'tp1_timestamp_diff_minutes': tp1_diff,
                'tp2_match': tp2_match,
                'tp2_timestamp_diff_minutes': tp2_diff,
                'tp3_match': tp3_match,
                'tp3_timestamp_diff_minutes': tp3_diff,
                'sl_match': sl_match,
                'sl_timestamp_diff_minutes': sl_diff,
                'status_match': status_match,
                'discrepancy_type': discrepancy_type,
                'discrepancy_severity': discrepancy_severity,
                'discrepancy_details': discrepancy_details,
                'validation_date': datetime.now(self.tz).isoformat()
            }
            
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
    
    def _compare_hit(
        self,
        auto_hit: bool,
        auto_datetime: Optional[str],
        manual_hit: bool,
        manual_datetime: Optional[str]
    ) -> Tuple[bool, Optional[int]]:
        """
        Compare hit status and timestamps.
        
        Returns:
            Tuple of (match, timestamp_diff_minutes)
        """
        # If hit status doesn't match, it's a mismatch
        if auto_hit != manual_hit:
            return False, None
        
        # If both are False, it's a match
        if not auto_hit and not manual_hit:
            return True, 0
        
        # If both are True, compare timestamps
        if auto_hit and manual_hit:
            if not auto_datetime or not manual_datetime:
                # One is missing timestamp - minor discrepancy
                return True, None
            
            try:
                # Parse timestamps
                if isinstance(auto_datetime, str):
                    auto_dt = datetime.fromisoformat(auto_datetime.replace('Z', '+00:00'))
                else:
                    auto_dt = auto_datetime
                
                if isinstance(manual_datetime, str):
                    manual_dt = datetime.fromisoformat(manual_datetime.replace('Z', '+00:00'))
                else:
                    manual_dt = manual_datetime
                
                # Ensure timezone aware
                if auto_dt.tzinfo is None:
                    auto_dt = self.tz.localize(auto_dt)
                if manual_dt.tzinfo is None:
                    manual_dt = self.tz.localize(manual_dt)
                
                # Calculate difference
                diff = abs((auto_dt - manual_dt).total_seconds() / 60.0)  # minutes
                
                # Check if within tolerance
                if diff <= self.timestamp_tolerance.total_seconds() / 60.0:
                    return True, int(diff)
                else:
                    return False, int(diff)
                    
            except Exception:
                return True, None  # Can't parse, assume match
        
        return True, 0
    
    def _determine_discrepancy(
        self,
        tp1_match: bool,
        tp2_match: bool,
        tp3_match: bool,
        sl_match: bool,
        status_match: bool
    ) -> Tuple[str, str]:
        """
        Determine discrepancy type and severity.
        
        Returns:
            Tuple of (discrepancy_type, discrepancy_severity)
        """
        all_match = tp1_match and tp2_match and tp3_match and sl_match and status_match
        
        if all_match:
            return 'NO_MISMATCH', 'MINOR'
        
        # Check for critical mismatches
        if not sl_match:
            return 'SL_MISMATCH', 'CRITICAL'
        
        if not status_match:
            return 'STATUS_MISMATCH', 'CRITICAL'
        
        # Check for TP mismatches
        if not tp1_match or not tp2_match or not tp3_match:
            return 'TP_MISMATCH', 'WARNING'
        
        # Check for timestamp mismatches only
        return 'TIMESTAMP_MISMATCH', 'MINOR'
    
    def validate_from_excel(
        self,
        excel_file_path: str,
        provider_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Validate signals by reading manual analysis from Excel file.
        
        Args:
            excel_file_path: Path to Excel file with manual analysis
            provider_name: Optional provider name filter
            
        Returns:
            List of validation results
        """
        try:
            # Read Excel file
            df = pd.read_excel(excel_file_path)
            
            # Expected columns: Signal ID, TP1 Hit, TP1 Hit Date, TP2 Hit, TP2 Hit Date, etc.
            # Also supports: Manual TP1 Hit, Manual TP1 Hit Date (from exported Excel)
            # Map Excel columns to our format
            column_mapping = {
                'Signal ID': 'signal_id',
                'signal_id': 'signal_id',
                'ID': 'signal_id',
                'id': 'signal_id',
                # Automated columns (for reference, not used in validation)
                'Auto TP1 Hit': 'tp1_hit',
                'Auto TP1 Hit Date': 'tp1_hit_datetime',
                'Auto TP2 Hit': 'tp2_hit',
                'Auto TP2 Hit Date': 'tp2_hit_datetime',
                'Auto TP3 Hit': 'tp3_hit',
                'Auto TP3 Hit Date': 'tp3_hit_datetime',
                'Auto SL Hit': 'sl_hit',
                'Auto SL Hit Date': 'sl_hit_datetime',
                'Auto Final Status': 'final_status',
                # Manual columns (these are what we validate)
                'Manual TP1 Hit': 'tp1_hit',
                'Manual TP1 Hit Date': 'tp1_hit_datetime',
                'TP1 Hit': 'tp1_hit',
                'tp1_hit': 'tp1_hit',
                'TP1 Hit Date': 'tp1_hit_datetime',
                'tp1_hit_datetime': 'tp1_hit_datetime',
                'Manual TP2 Hit': 'tp2_hit',
                'Manual TP2 Hit Date': 'tp2_hit_datetime',
                'TP2 Hit': 'tp2_hit',
                'tp2_hit': 'tp2_hit',
                'TP2 Hit Date': 'tp2_hit_datetime',
                'tp2_hit_datetime': 'tp2_hit_datetime',
                'Manual TP3 Hit': 'tp3_hit',
                'Manual TP3 Hit Date': 'tp3_hit_datetime',
                'TP3 Hit': 'tp3_hit',
                'tp3_hit': 'tp3_hit',
                'TP3 Hit Date': 'tp3_hit_datetime',
                'tp3_hit_datetime': 'tp3_hit_datetime',
                'Manual SL Hit': 'sl_hit',
                'Manual SL Hit Date': 'sl_hit_datetime',
                'SL Hit': 'sl_hit',
                'sl_hit': 'sl_hit',
                'SL Hit Date': 'sl_hit_datetime',
                'sl_hit_datetime': 'sl_hit_datetime',
                'Manual Final Status': 'final_status',
                'Status': 'final_status',
                'status': 'final_status',
                'Final Status': 'final_status'
            }
            
            # Normalize column names
            df.columns = df.columns.str.strip()
            
            # Map columns
            manual_results = []
            for _, row in df.iterrows():
                manual_result = {}
                
                # Get signal_id
                signal_id = None
                for col in ['Signal ID', 'signal_id', 'ID', 'id']:
                    if col in df.columns:
                        signal_id = row.get(col)
                        if pd.notna(signal_id):
                            signal_id = int(signal_id)
                            break
                
                if not signal_id:
                    continue
                
                # Map other fields - prioritize "Manual" columns over generic ones
                # First pass: collect all Manual columns
                manual_columns = {}
                for excel_col, our_field in column_mapping.items():
                    if excel_col.startswith('Manual ') and excel_col in df.columns:
                        value = row.get(excel_col)
                        if pd.notna(value) and value != '':
                            manual_columns[our_field] = (excel_col, value)
                
                # Second pass: use Manual columns if available, otherwise use generic columns
                for excel_col, our_field in column_mapping.items():
                    # Skip if we already have a Manual column for this field
                    if our_field in manual_columns:
                        excel_col, value = manual_columns[our_field]
                    elif excel_col in df.columns:
                        value = row.get(excel_col)
                    else:
                        continue
                    
                    # Skip Auto columns (we only want Manual data for validation)
                    if excel_col.startswith('Auto '):
                        continue
                    
                    if pd.notna(value) and value != '':
                        # Convert boolean columns (TP1 Hit, TP2 Hit, etc.)
                        if 'hit' in our_field.lower() and 'datetime' not in our_field:
                            if isinstance(value, bool):
                                manual_result[our_field] = value
                            elif isinstance(value, str):
                                manual_result[our_field] = value.lower() in ['true', 'yes', '1', 'hit', 'y']
                            else:
                                manual_result[our_field] = bool(value)
                        # Convert datetime columns
                        elif 'datetime' in our_field or 'date' in our_field:
                            if isinstance(value, pd.Timestamp):
                                manual_result[our_field] = value.isoformat()
                            elif isinstance(value, str) and value.strip():
                                manual_result[our_field] = value
                            elif value:
                                manual_result[our_field] = str(value)
                        else:
                            manual_result[our_field] = value
                
                manual_results.append({
                    'signal_id': signal_id,
                    'manual_result': manual_result
                })
            
            # Fetch automated results and validate
            validation_results = []
            for item in manual_results:
                signal_id = item['signal_id']
                manual_result = item['manual_result']
                
                # Fetch automated result
                automated_result = self._get_automated_result(signal_id, provider_name)
                if not automated_result:
                    continue
                
                # Validate
                validation = self.validate_signal(
                    signal_id=signal_id,
                    automated_result=automated_result,
                    manual_result=manual_result
                )
                
                validation['manual_analysis_source'] = 'EXCEL'
                validation['manual_analysis_date'] = datetime.now(self.tz).isoformat()
                
                validation_results.append(validation)
            
            return validation_results
            
        except Exception as e:
            return [{"error": f"Excel validation failed: {str(e)}"}]
    
    def _get_automated_result(
        self,
        signal_id: int,
        provider_name: Optional[str] = None
    ) -> Optional[Dict]:
        """Fetch automated analysis result from database."""
        if not self.supabase:
            return None
        
        try:
            query = self.supabase.table('analysis_results').select('*')
            query = query.eq('signal_id', signal_id)
            query = query.eq('analysis_method', 'automated')
            
            if provider_name:
                query = query.eq('provider_name', provider_name)
            
            result = query.execute()
            if result.data:
                return result.data[0]
            return None
            
        except Exception:
            return None
    
    def save_validation_report(self, validation: Dict) -> bool:
        """
        Save validation report to database.
        
        Args:
            validation: Validation result dictionary
            
        Returns:
            True if saved successfully
        """
        if not self.supabase or 'error' in validation:
            return False
        
        try:
            record = {
                'signal_id': validation.get('signal_id'),
                'analysis_result_id': validation.get('analysis_result_id'),
                'tp1_match': validation.get('tp1_match'),
                'tp1_timestamp_diff_minutes': validation.get('tp1_timestamp_diff_minutes'),
                'tp2_match': validation.get('tp2_match'),
                'tp2_timestamp_diff_minutes': validation.get('tp2_timestamp_diff_minutes'),
                'tp3_match': validation.get('tp3_match'),
                'tp3_timestamp_diff_minutes': validation.get('tp3_timestamp_diff_minutes'),
                'sl_match': validation.get('sl_match'),
                'sl_timestamp_diff_minutes': validation.get('sl_timestamp_diff_minutes'),
                'status_match': validation.get('status_match'),
                'discrepancy_type': validation.get('discrepancy_type'),
                'discrepancy_severity': validation.get('discrepancy_severity'),
                'discrepancy_details': validation.get('discrepancy_details'),
                'manual_analysis_source': validation.get('manual_analysis_source'),
                'manual_analysis_date': validation.get('manual_analysis_date'),
                'validation_date': validation.get('validation_date')
            }
            
            # Remove None values
            record = {k: v for k, v in record.items() if v is not None}
            
            self.supabase.table('validation_reports').insert(record).execute()
            return True
            
        except Exception as e:
            print(f"Error saving validation report: {e}")
            return False
    
    def get_validation_summary(
        self,
        provider_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get validation summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.supabase:
            return {"error": "Database not available"}
        
        try:
            query = self.supabase.table('validation_reports').select('*')
            
            if provider_name:
                # Join with analysis_results to filter by provider
                # For simplicity, fetch all and filter
                pass
            
            if start_date:
                query = query.gte('validation_date', start_date.isoformat())
            if end_date:
                query = query.lte('validation_date', end_date.isoformat())
            
            result = query.execute()
            reports = result.data if result.data else []
            
            if not reports:
                return {
                    'total_validated': 0,
                    'matches': 0,
                    'discrepancies': 0,
                    'match_rate': 0.0
                }
            
            total = len(reports)
            matches = sum(1 for r in reports if r.get('discrepancy_type') == 'NO_MISMATCH')
            discrepancies = total - matches
            
            # Count by severity
            critical = sum(1 for r in reports if r.get('discrepancy_severity') == 'CRITICAL')
            warning = sum(1 for r in reports if r.get('discrepancy_severity') == 'WARNING')
            minor = sum(1 for r in reports if r.get('discrepancy_severity') == 'MINOR')
            
            # Count by type
            tp_mismatch = sum(1 for r in reports if r.get('discrepancy_type') == 'TP_MISMATCH')
            sl_mismatch = sum(1 for r in reports if r.get('discrepancy_type') == 'SL_MISMATCH')
            timestamp_mismatch = sum(1 for r in reports if r.get('discrepancy_type') == 'TIMESTAMP_MISMATCH')
            
            return {
                'total_validated': total,
                'matches': matches,
                'discrepancies': discrepancies,
                'match_rate': round((matches / total * 100) if total > 0 else 0, 2),
                'by_severity': {
                    'critical': critical,
                    'warning': warning,
                    'minor': minor
                },
                'by_type': {
                    'tp_mismatch': tp_mismatch,
                    'sl_mismatch': sl_mismatch,
                    'timestamp_mismatch': timestamp_mismatch
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to get summary: {str(e)}"}

