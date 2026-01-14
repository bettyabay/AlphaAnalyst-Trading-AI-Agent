"""
Telegram Signal Parser
Parses trading signals from Telegram messages in various formats.
"""
import re
from datetime import datetime
from typing import Optional, Dict
import pytz


class TelegramSignalParser:
    """
    Parser for extracting trading signals from Telegram messages.
    Handles structured formats like:
    
    📣GBP/USD📣
    Direction: SELL
    Entry Price:   1.3493
    TP1     1.3478
    TP2     1.3458
    TP3     1.3426
    SL       1.3546
    """
    
    def __init__(self):
        self.gmt4_tz = pytz.timezone('Asia/Dubai')
        self.utc_tz = pytz.timezone('UTC')
    
    def parse(self, message_text: str) -> Optional[Dict]:
        """
        Main parsing function. Tries multiple parsing strategies.
        Skips non-signal messages (images, regular text, etc.)
        
        Args:
            message_text: Raw message text from Telegram
            
        Returns:
            Dict with parsed signal data or None if parsing fails or not a signal
        """
        if not message_text or not message_text.strip():
            return None
        
        # Quick validation: Must have minimum required elements for a signal
        text_upper = message_text.upper()
        
        # Must have either BUY/SELL or Direction keyword
        has_direction = (
            'BUY' in text_upper or 'SELL' in text_upper or 
            'DIRECTION' in text_upper
        )
        
        # Must have a currency pair or symbol
        has_symbol = (
            re.search(r'[A-Z]{3,4}/[A-Z]{3,4}', text_upper) or  # EUR/USD format
            re.search(r'[A-Z]{6,7}', text_upper) or  # EURUSD format
            '📣' in message_text  # Signal emoji
        )
        
        # Must have at least one price (decimal number)
        has_price = bool(re.search(r'\d+\.\d{2,6}', message_text))
        
        # Skip if doesn't meet minimum requirements
        if not (has_direction and has_symbol and has_price):
            return None
        
        # Try structured format parsing first (most common)
        parsed = self._parse_structured_format(message_text)
        if parsed:
            return parsed
        
        # Try alternative formats
        parsed = self._parse_compact_format(message_text)
        if parsed:
            return parsed
        
        # Try natural language parsing (basic)
        parsed = self._parse_natural_language(message_text)
        if parsed:
            return parsed
        
        return None
    
    def _parse_structured_format(self, text: str) -> Optional[Dict]:
        """
        Parse structured format like:
        📣GBP/USD📣
        Direction: SELL
        Entry Price:   1.3493
        TP1     1.3478
        TP2     1.3458
        TP3     1.3426
        SL       1.3546
        """
        try:
            # Extract symbol - try multiple patterns
            symbol = None
            
            # Pattern 1: Between 📣 emojis
            symbol_match = re.search(r'📣\s*([A-Z]+/[A-Z]+|[A-Z]+)\s*📣', text, re.IGNORECASE)
            if symbol_match:
                symbol = symbol_match.group(1).strip().upper()
            
            # Pattern 2: After 📣 emoji (no closing)
            if not symbol:
                symbol_match = re.search(r'📣\s*([A-Z]+/[A-Z]+|[A-Z]+)', text, re.IGNORECASE)
                if symbol_match:
                    symbol = symbol_match.group(1).strip().upper()
            
            # Pattern 3: Currency pairs anywhere (EUR/USD, GBPUSD, etc.)
            if not symbol:
                symbol_match = re.search(r'\b([A-Z]{3,4}/[A-Z]{3,4}|[A-Z]{6,7})\b', text, re.IGNORECASE)
                if symbol_match:
                    potential = symbol_match.group(1).strip().upper()
                    # Validate it looks like a currency pair
                    if '/' in potential or len(potential) >= 6:
                        symbol = potential
            
            if not symbol:
                return None
            
            symbol = self.normalize_symbol(symbol)
            
            # Extract direction (BUY/SELL) - more flexible
            action = None
            
            # Pattern 1: "Direction: SELL" or "Direction:BUY"
            direction_match = re.search(r'Direction[:\s]*([BUYSELL]+)', text, re.IGNORECASE)
            if direction_match:
                action = direction_match.group(1).strip().lower()
            
            # Pattern 2: Standalone BUY/SELL (but not part of another word)
            if not action:
                direction_match = re.search(r'\b(BUY|SELL)\b', text, re.IGNORECASE)
                if direction_match:
                    action = direction_match.group(1).strip().lower()
            
            if not action or action not in ['buy', 'sell']:
                return None
            
            # Extract Entry Price - more flexible patterns
            entry_price = None
            
            # Pattern 1: "Entry Price: 1.3493" or "Entry: 1.3493"
            entry_match = re.search(r'Entry\s*(?:Price)?[:\s]*([\d.]+)', text, re.IGNORECASE)
            if entry_match:
                try:
                    entry_price = float(entry_match.group(1).strip())
                except ValueError:
                    pass
            
            # Pattern 2: "@ 1.3493" or "at 1.3493"
            if not entry_price:
                entry_match = re.search(r'[@\s]+([\d.]+)', text, re.IGNORECASE)
                if entry_match:
                    try:
                        entry_price = float(entry_match.group(1).strip())
                    except ValueError:
                        pass
            
            # Pattern 3: First price-like number after symbol/direction
            if not entry_price:
                # Find all numbers with decimals
                price_matches = re.findall(r'\b(\d+\.\d{2,6})\b', text)
                if price_matches:
                    try:
                        entry_price = float(price_matches[0])
                    except (ValueError, IndexError):
                        pass
            
            if not entry_price:
                return None
            
            # Extract Stop Loss - more flexible
            stop_loss = None
            sl_match = re.search(r'SL[:\s]+([\d.]+)', text, re.IGNORECASE)
            if not sl_match:
                sl_match = re.search(r'Stop\s*Loss[:\s]+([\d.]+)', text, re.IGNORECASE)
            if sl_match:
                try:
                    stop_loss = float(sl_match.group(1).strip())
                except ValueError:
                    pass
            
            # Extract Take Profit targets - more flexible
            tp_matches = re.findall(r'TP(\d+)[:\s]+([\d.]+)', text, re.IGNORECASE)
            if not tp_matches:
                # Try "Target 1:", "Target1:", etc.
                tp_matches = re.findall(r'Target\s*(\d+)[:\s]+([\d.]+)', text, re.IGNORECASE)
            
            targets = {}
            for tp_num, tp_value in tp_matches:
                try:
                    targets[f'target_{tp_num}'] = float(tp_value.strip())
                except ValueError:
                    continue
            
            # Sort targets by number
            sorted_targets = {}
            for i in range(1, 6):  # Support up to TP5
                key = f'target_{i}'
                if key in targets:
                    sorted_targets[key] = targets[key]
            
            # Build result
            result = {
                "symbol": symbol,
                "action": action,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "signal_date": datetime.now(self.gmt4_tz).isoformat(),
            }
            
            # Add targets
            for i, (key, value) in enumerate(sorted_targets.items(), 1):
                result[f"target_{i}"] = value
            
            return result
            
        except Exception as e:
            print(f"Error parsing structured format: {e}")
            return None
    
    def _parse_compact_format(self, text: str) -> Optional[Dict]:
        """
        Parse compact format like:
        BUY EURUSD @ 1.0850 SL: 1.0800 TP1: 1.0900 TP2: 1.0950
        """
        try:
            # Pattern: BUY/SELL SYMBOL @ ENTRY SL: STOP TP1: TARGET1 TP2: TARGET2
            pattern = r'(BUY|SELL)\s+([A-Z]+[/]?[A-Z]*)\s+@\s+([\d.]+)\s+SL[:\s]+([\d.]+)(?:\s+TP\d+[:\s]+([\d.]+))*(?:\s+TP\d+[:\s]+([\d.]+))*(?:\s+TP\d+[:\s]+([\d.]+))*'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if not match:
                return None
            
            action = match.group(1).strip().lower()
            symbol = self.normalize_symbol(match.group(2).strip().upper())
            entry_price = float(match.group(3).strip())
            stop_loss = float(match.group(4).strip())
            
            # Extract all TP values
            tp_pattern = r'TP\d+[:\s]+([\d.]+)'
            tp_matches = re.findall(tp_pattern, text, re.IGNORECASE)
            
            result = {
                "symbol": symbol,
                "action": action,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "signal_date": datetime.now(self.gmt4_tz).isoformat(),
            }
            
            for i, tp_value in enumerate(tp_matches, 1):
                result[f"target_{i}"] = float(tp_value.strip())
            
            return result
            
        except Exception as e:
            print(f"Error parsing compact format: {e}")
            return None
    
    def _parse_natural_language(self, text: str) -> Optional[Dict]:
        """
        Basic natural language parsing (can be enhanced with LLM later).
        Looks for key phrases and numbers.
        """
        try:
            # Look for BUY/SELL
            action_match = re.search(r'\b(BUY|SELL)\b', text, re.IGNORECASE)
            if not action_match:
                return None
            
            action = action_match.group(1).strip().lower()
            
            # Look for currency pairs or symbols
            symbol_match = re.search(r'\b([A-Z]{3,6}[/]?[A-Z]{3,6}|[A-Z]{2,6})\b', text)
            if not symbol_match:
                return None
            
            symbol = self.normalize_symbol(symbol_match.group(1).strip().upper())
            
            # Look for price patterns (numbers with decimals)
            prices = re.findall(r'\b(\d+\.\d{2,6})\b', text)
            if len(prices) < 2:  # Need at least entry and stop loss
                return None
            
            # Assume first price is entry, second is stop loss
            entry_price = float(prices[0])
            stop_loss = float(prices[1])
            
            result = {
                "symbol": symbol,
                "action": action,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "signal_date": datetime.now(self.gmt4_tz).isoformat(),
            }
            
            # Additional prices might be targets
            for i, price in enumerate(prices[2:], 1):
                result[f"target_{i}"] = float(price)
            
            return result
            
        except Exception as e:
            print(f"Error parsing natural language: {e}")
            return None
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format to match database format.
        Examples:
        - "GBP/USD" -> "C:GBPUSD"
        - "EURUSD" -> "C:EURUSD"
        - "XAUUSD" -> "C:XAUUSD" (if commodity)
        - "SPX" -> "^SPX" (if index)
        """
        # Remove spaces and special characters
        symbol = symbol.replace('/', '').replace('-', '').replace('_', '').strip().upper()
        
        # Check if it's a currency pair (typically 6-7 chars: EURUSD, GBPUSD, etc.)
        if len(symbol) >= 6 and len(symbol) <= 7:
            # Common currency pairs
            currency_pairs = [
                'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
                'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURCHF', 'AUDNZD', 'EURAUD',
                'XAUUSD', 'XAGUSD', 'XPDUSD', 'XPTUSD'  # Precious metals
            ]
            
            if symbol in currency_pairs or symbol.startswith('XAU') or symbol.startswith('XAG'):
                return f"C:{symbol}"
        
        # Check if it's a known commodity
        commodities = ['GOLD', 'SILVER', 'OIL', 'WTI', 'BRENT']
        if symbol in commodities:
            if symbol == 'GOLD':
                return "C:XAUUSD"
            elif symbol == 'SILVER':
                return "C:XAGUSD"
            elif symbol in ['OIL', 'WTI', 'BRENT']:
                return f"C:{symbol}"
        
        # Check if it's an index
        indices = ['SPX', 'DJI', 'NDX', 'RUT']
        if symbol in indices:
            return f"^{symbol}"
        
        # Default: assume currency pair if 6-7 chars
        if len(symbol) >= 6 and len(symbol) <= 7:
            return f"C:{symbol}"
        
        # Return as-is for stocks and other symbols
        return symbol


