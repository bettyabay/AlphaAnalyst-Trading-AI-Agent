"""
Test script for Telegram Signal Parser
Tests the parser with the example message format provided.
"""
import sys
import io
from tradingagents.dataflows.telegram_signal_parser import TelegramSignalParser

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Example message from user
test_message = """📣GBP/USD📣

Direction: SELL

Entry Price:   1.3493

TP1     1.3478
TP2     1.3458
TP3     1.3426

SL       1.3546"""

def test_parser():
    parser = TelegramSignalParser()
    
    print("=" * 60)
    print("Testing Telegram Signal Parser")
    print("=" * 60)
    print("\nInput message:")
    try:
        print(test_message)
    except UnicodeEncodeError:
        print(test_message.encode('ascii', 'ignore').decode('ascii'))
    print("\n" + "=" * 60)
    
    # Parse the message
    result = parser.parse(test_message)
    
    if result:
        print("✅ Parsing successful!")
        print("\nParsed signal data:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # Verify expected values
        expected = {
            "symbol": "C:GBPUSD",
            "action": "sell",
            "entry_price": 1.3493,
            "stop_loss": 1.3546,
            "target_1": 1.3478,
            "target_2": 1.3458,
            "target_3": 1.3426
        }
        
        print("\n" + "=" * 60)
        print("Validation:")
        all_match = True
        for key, expected_value in expected.items():
            actual_value = result.get(key)
            if actual_value == expected_value:
                print(f"  ✅ {key}: {actual_value}")
            else:
                print(f"  ❌ {key}: Expected {expected_value}, Got {actual_value}")
                all_match = False
        
        if all_match:
            print("\n🎉 All validations passed!")
        else:
            print("\n⚠️ Some validations failed")
    else:
        print("❌ Parsing failed - returned None")
        print("\nTrying alternative formats...")
        
        # Test with variations
        variations = [
            "BUY EURUSD @ 1.0850 SL: 1.0800 TP1: 1.0900 TP2: 1.0950",
            "SELL GBPUSD Entry: 1.3500 Stop Loss: 1.3550 Target: 1.3450",
        ]
        
        for var_msg in variations:
            print(f"\nTesting: {var_msg[:50]}...")
            result = parser.parse(var_msg)
            if result:
                print(f"  ✅ Parsed: {result.get('action')} {result.get('symbol')} @ {result.get('entry_price')}")
            else:
                print(f"  ❌ Failed to parse")

if __name__ == "__main__":
    test_parser()

