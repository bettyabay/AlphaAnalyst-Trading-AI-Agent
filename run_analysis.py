"""
Simple script to run signal analysis
Run with: python run_analysis.py
"""
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from tradingagents.dataflows.run_signal_analysis import (
    run_analysis_for_all_signals,
    run_backtest_for_all_signals,
    generate_daily_report_and_save,
    calculate_provider_metrics
)

if __name__ == "__main__":
    print("=" * 60)
    print("Signal Analysis Batch Runner")
    print("=" * 60)
    
    # Generate daily report
    print("\n1. Generating daily report...")
    try:
        daily_report = generate_daily_report_and_save()
        if 'error' not in daily_report:
            print(f"   Report generated for: {daily_report.get('log_date', 'N/A')}")
            print(f"   Currencies: {daily_report.get('currencies_ingested', 0)}/{daily_report.get('currencies_total', 28)}")
            print(f"   Signals Analyzed: {daily_report.get('signals_analyzed', 0)}")
        else:
            print(f"   Error: {daily_report.get('error')}")
    except Exception as e:
        print(f"   Error generating report: {str(e)}")
    
    # Run analysis (uncomment to run)
    print("\n2. Running signal analysis...")
    print("   (This may take a while. Uncomment in script to run.)")
    # try:
    #     analysis_summary = run_analysis_for_all_signals()
    #     print(f"   Total Signals: {analysis_summary.get('total_signals', 0)}")
    #     print(f"   Analyzed: {analysis_summary.get('analyzed', 0)}")
    #     print(f"   Errors: {analysis_summary.get('errors', 0)}")
    #     print(f"   Success Rate: {analysis_summary.get('success_rate', 0):.2f}%")
    # except Exception as e:
    #     print(f"   Error: {str(e)}")
    
    print("\nDone!")

