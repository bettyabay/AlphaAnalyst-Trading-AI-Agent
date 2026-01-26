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
    
    def _validate_signal(self, signal_data: Dict) -> bool:
        """
        Validate that a signal has all required fields.
        
        Required fields:
        - symbol: Must be present and valid
        - action: Must be 'buy' or 'sell'
        - entry_price: Must be a valid number
        - stop_loss: Must be a valid number
        - At least one target (target_1, target_2, etc.)
        
        Args:
            signal_data: Dictionary with parsed signal data
            
        Returns:
            True if signal is valid and complete, False otherwise
        """
        if not signal_data:
            return False
        
        # Check required fields
        if not signal_data.get('symbol'):
            return False
        
        if signal_data.get('action') not in ['buy', 'sell']:
            return False
        
        entry_price = signal_data.get('entry_price')
        if entry_price is None or not isinstance(entry_price, (int, float)):
            return False
        
        # Reject single-digit or very small integers as entry prices (likely not a real price)
        # Forex prices are typically 1.0+ (like 1.1240) or 100+ (like 4595.92 for XAUUSD)
        # Reject anything less than 0.1 or single-digit integers less than 10
        if isinstance(entry_price, (int, float)):
            if entry_price < 0.1:
                return False
            # Reject single-digit integers (1-9) as they're likely not prices
            if isinstance(entry_price, int) and 1 <= entry_price <= 9:
                return False
            # Reject if it's a float but less than 1.0 (unless it's a very small decimal like 0.95)
            if isinstance(entry_price, float) and 0.1 <= entry_price < 1.0:
                # Allow small decimals like 0.95, 0.85 (some forex pairs can be below 1.0)
                # But reject if it's clearly an integer like 3.0
                if entry_price == int(entry_price) and entry_price < 10:
                    return False
        
        stop_loss = signal_data.get('stop_loss')
        if stop_loss is None or not isinstance(stop_loss, (int, float)):
            return False
        
        # Check for at least one target
        has_target = False
        for i in range(1, 6):
            target = signal_data.get(f'target_{i}')
            if target is not None and isinstance(target, (int, float)):
                has_target = True
                break
        
        if not has_target:
            return False
        
        return True
    
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
        
        # Must have a currency pair or symbol (including indices with numbers like NAS100, US30)
        # US30 is 4 chars, NAS100 is 6 chars, so we need to support 3-8 char symbols
        has_symbol = (
            re.search(r'[A-Z]{3,4}/[A-Z]{3,4}', text_upper) or  # EUR/USD format
            re.search(r'[A-Z0-9]{3,8}', text_upper) or  # EURUSD, BTCUSD, NAS100, US30 format (3-8 chars)
            re.search(r'[A-Z]{6,8}', text_upper) or  # EURUSD, BTCUSD format (fallback, letters only)
            re.search(r'[a-z0-9]{3,8}', message_text.lower()) or  # xauusd, nas100, us30 format (lowercase)
            re.search(r'[a-z]{6,8}', message_text.lower()) or  # xauusd format (lowercase, fallback)
            '📣' in message_text  # Signal emoji
        )
        
        # Must have at least one price (decimal or integer number)
        has_price = bool(re.search(r'\d+\.\d{2,6}|\d{4,}', message_text))
        
        # Skip if doesn't meet minimum requirements
        if not (has_direction and has_symbol and has_price):
            return None
        
        # Try SIGNAL ALERT format with emojis FIRST (very specific patterns)
        # This handles "SIGNAL ALERT\nBUY XAUUSD..." and "Gold Sell Now Scalping @ ..." formats
        if 'SIGNAL ALERT' in message_text.upper() or 'SCALPING' in message_text.upper() or ('GOLD' in message_text.upper() and '@' in message_text):
            parsed = self._parse_signal_alert_format(message_text)
            if parsed and self._validate_signal(parsed):
                return parsed
        
        # Try range format with explanation text (very specific pattern)
        # This handles "at any price between X until Y" format
        # If this pattern is detected, ONLY try this parser to avoid incorrect matches
        if 'at any price between' in message_text.lower() or ('between' in message_text.lower() and 'until' in message_text.lower()):
            parsed = self._parse_range_with_explanation(message_text)
            if parsed and self._validate_signal(parsed):
                return parsed
            # If range format parser fails, still try other parsers as fallback
        
        # Try range format: "at any price between X till Y" or "Buy EURUSD at any price between X till Y" (Format 3)
        # This handles formats like:
        # - Buy EURUSD at any price between 1.1695 till 1.1670
        # - Target 1: 1.1742, Target 2: 1.1820, etc.
        if 'between' in message_text.lower() and ('till' in message_text.lower() or 'until' in message_text.lower() or 'and' in message_text.lower()):
            parsed = self._parse_range_format(message_text)
            if parsed and self._validate_signal(parsed):
                return parsed
        
        # Skip structured format if message contains range-like patterns to avoid incorrect matches
        has_range_pattern = (
            'between' in message_text.lower() and 
            ('until' in message_text.lower() or 'till' in message_text.lower() or 'and' in message_text.lower())
        )
        
        # Try structured format parsing (most common - PipXpert format)
        # But skip if message has range patterns (already tried range parsers above)
        if not has_range_pattern:
            parsed = self._parse_structured_format(message_text)
            if parsed and self._validate_signal(parsed):
                return parsed
        
        # Try simple format: SYMBOL ACTION ENTRY_PRICE with SL: and TP: (Format for NAS100, US30, etc.)
        parsed = self._parse_simple_format(message_text)
        if parsed and self._validate_signal(parsed):
            return parsed
        
        # Try inline format: SYMBOL ACTION ENTRY (Format 1 & 2)
        parsed = self._parse_inline_format(message_text)
        if parsed and self._validate_signal(parsed):
            return parsed
        
        # Try alternative formats
        parsed = self._parse_compact_format(message_text)
        if parsed and self._validate_signal(parsed):
            return parsed
        
        # Try emoji format: XAUUSD BUY_ 4595 _ 92 with TP¹, TP², TP³
        parsed = self._parse_emoji_format(message_text)
        if parsed and self._validate_signal(parsed):
            return parsed
        
        # Try natural language parsing (basic)
        parsed = self._parse_natural_language(message_text)
        if parsed and self._validate_signal(parsed):
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
            
            # Pattern 3: Currency pairs anywhere (EUR/USD, GBPUSD, BTCUSD, etc.)
            if not symbol:
                # First, try to find known trading symbol patterns
                # Look for currency pairs with slash
                symbol_match = re.search(r'\b([A-Z]{3}/[A-Z]{3})\b', text, re.IGNORECASE)
                if symbol_match:
                    potential = symbol_match.group(1).strip().upper()
                    if self._is_valid_trading_symbol(potential.replace('/', '')):
                        symbol = potential
                
                # If not found, look for symbols with numbers (indices like NAS100, US30) or 6-7 char currency pairs
                if not symbol:
                    # First try symbols with numbers (indices like NAS100, US30)
                    symbol_matches = re.findall(r'\b([A-Z0-9]{3,8})\b', text, re.IGNORECASE)
                    for potential in symbol_matches:
                        potential = potential.strip().upper()
                        if self._is_valid_trading_symbol(potential):
                            symbol = potential
                            break
                    
                    # If not found, try 6-7 char currency pairs (letters only)
                    if not symbol:
                        symbol_matches = re.findall(r'\b([A-Z]{6,7})\b', text, re.IGNORECASE)
                        for potential in symbol_matches:
                            potential = potential.strip().upper()
                            if self._is_valid_trading_symbol(potential):
                                symbol = potential
                                break
                
                # Last resort: 8 char symbols (some crypto pairs)
                if not symbol:
                    symbol_matches = re.findall(r'\b([A-Z]{8})\b', text, re.IGNORECASE)
                    for potential in symbol_matches:
                        potential = potential.strip().upper()
                        if self._is_valid_trading_symbol(potential):
                            symbol = potential
                            break
            
            if not symbol:
                return None
            
            try:
                symbol = self.normalize_symbol(symbol)
            except ValueError:
                # Invalid symbol - skip this signal
                return None
            
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
            
            # Pattern 2: "@ 1.3493" or "at 1.3493" (but only match actual prices, not small integers)
            if not entry_price:
                # Match "@" or "at" followed by a price (decimal with 2+ places or 4+ digit integer)
                entry_match = re.search(r'[@\s]at\s+([\d.]+)', text, re.IGNORECASE)
                if entry_match:
                    potential_price = entry_match.group(1).strip()
                    # Only accept if it's a decimal with 2+ places or 4+ digit integer
                    if re.match(r'\d+\.\d{2,6}|\d{4,}', potential_price):
                        # Check it's not in a context like "3 Take Profit" or "3 trades"
                        price_pattern = re.escape(potential_price)
                        context_patterns = [
                            rf'\b{price_pattern}\s+(?:Take\s+Profit|trades?|trade|targets?|position|minutes?)',
                            rf'(?:Take\s+Profit|trades?|trade|targets?|position|minutes?).*?\b{price_pattern}',
                        ]
                        is_price = True
                        for ctx_pattern in context_patterns:
                            if re.search(ctx_pattern, text, re.IGNORECASE):
                                is_price = False
                                break
                        if is_price:
                            try:
                                entry_price = float(potential_price)
                            except ValueError:
                                pass
            
            # Pattern 3: First price-like number after symbol/direction (decimal or integer)
            # But avoid matching small integers that are clearly not prices (like "3" from "3 Take Profit Targets")
            if not entry_price:
                # Find all numbers (decimals with 2+ decimal places, or integers with 4+ digits)
                # Exclude numbers that are clearly not prices (like "3" from "3 Take Profit", "3 trades", etc.)
                price_matches = re.findall(r'\b(\d+\.\d{2,6}|\d{4,})\b', text)
                
                # Filter out numbers that are clearly not prices
                # Skip numbers that appear in context like "3 Take Profit", "3 trades", "1st trade", etc.
                filtered_prices = []
                for price_str in price_matches:
                    # Check if this number appears in a context that suggests it's not a price
                    price_pattern = re.escape(price_str)
                    # Look for context that suggests it's not a price
                    context_patterns = [
                        rf'\b{price_pattern}\s+(?:Take\s+Profit|trades?|trade|targets?|position|minutes?|quiz)',
                        rf'(?:Take\s+Profit|trades?|trade|targets?|position|minutes?|quiz).*?\b{price_pattern}',
                        rf'\b(?:1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)\s+.*?\b{price_pattern}',
                        rf'\btrade\s+{price_pattern}',  # "trade 1", "trade 2", "trade 3"
                        rf'\b{price_pattern}\s*%',  # Percentage
                        rf'Risk\s+{price_pattern}',  # Risk percentage
                        rf'How\s+to\s+.*?\b{price_pattern}',  # "How to trade the signal with 3..."
                        rf'signal\s+with\s+{price_pattern}',  # "signal with 3 targets"
                        rf'place\s+{price_pattern}',  # "place 3 trades"
                        rf'open\s+{price_pattern}',  # "open 3 trade position"
                    ]
                    
                    is_price = True
                    for ctx_pattern in context_patterns:
                        if re.search(ctx_pattern, text, re.IGNORECASE):
                            is_price = False
                            break
                    
                    if is_price:
                        filtered_prices.append(price_str)
                
                if filtered_prices:
                    try:
                        entry_price = float(filtered_prices[0])
                    except (ValueError, IndexError):
                        pass
            
            if not entry_price:
                return None
            
            # Extract Stop Loss - more flexible
            stop_loss = None
            sl_patterns = [
                r'SL[:\s]+([\d.]+)',
                r'Stop\s*Loss[:\s]+([\d.]+)',
                r'Stop\s*Loss[._\s]+([\d.]+)',  # For "STOP LOSS ____ 4454" or "STOP LOSS ....4590"
            ]
            for pattern in sl_patterns:
                sl_match = re.search(pattern, text, re.IGNORECASE)
                if sl_match:
                    try:
                        stop_loss = float(sl_match.group(1).strip())
                        break
                    except ValueError:
                        continue
            
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
            try:
                symbol = self.normalize_symbol(match.group(2).strip().upper())
            except ValueError:
                # Invalid symbol - skip this signal
                return None
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
            
            # Look for currency pairs or symbols (avoid common words)
            # Try to find known trading symbol patterns first
            symbol_match = None
            
            # Pattern 1: Currency pairs with slash (EUR/USD, GBP/AUD)
            symbol_match = re.search(r'\b([A-Z]{3}/[A-Z]{3})\b', text)
            
            # Pattern 2: Currency pairs without slash (EURUSD, GBPAUD, BTCUSD) or indices with numbers (NAS100, US30)
            if not symbol_match:
                # Try symbols with numbers first (indices like NAS100, US30)
                symbol_match = re.search(r'\b([A-Z0-9]{3,8})\b', text)
                if symbol_match:
                    potential = symbol_match.group(1).strip().upper()
                    # Check against blacklist
                    blacklist = {'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 'DIRECTION', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'TP', 'SL'}
                    if potential in blacklist:
                        symbol_match = None
                    elif not self._is_valid_trading_symbol(potential):
                        symbol_match = None
                
                # Fallback to 6-7 char currency pairs (letters only)
                if not symbol_match:
                    symbol_match = re.search(r'\b([A-Z]{6,7})\b', text)
                    if symbol_match:
                        potential = symbol_match.group(1).strip().upper()
                        # Check against blacklist
                        blacklist = {'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 'DIRECTION'}
                        if potential in blacklist:
                            symbol_match = None
            
            if not symbol_match:
                return None
            
            try:
                symbol = self.normalize_symbol(symbol_match.group(1).strip().upper())
            except ValueError:
                # Invalid symbol - skip this signal
                return None
            
            # Look for price patterns (numbers with decimals or large integers)
            # Filter out numbers that are clearly not prices
            price_matches = re.findall(r'\b(\d+\.\d{2,6}|\d{4,})\b', text)
            
            # Filter out numbers that are clearly not prices
            filtered_prices = []
            for price_str in price_matches:
                # Check if this number appears in a context that suggests it's not a price
                price_pattern = re.escape(price_str)
                # Look for context that suggests it's not a price
                context_patterns = [
                    rf'\b{price_pattern}\s+(?:Take\s+Profit|trades?|trade|targets?|position)',
                    rf'(?:Take\s+Profit|trades?|trade|targets?|position).*?\b{price_pattern}',
                    rf'\b(?:1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)\s+.*?\b{price_pattern}',
                    rf'\b{price_pattern}\s*%',  # Percentage
                    rf'Risk\s+{price_pattern}',  # Risk percentage
                ]
                
                is_price = True
                for ctx_pattern in context_patterns:
                    if re.search(ctx_pattern, text, re.IGNORECASE):
                        is_price = False
                        break
                
                if is_price:
                    filtered_prices.append(price_str)
            
            if len(filtered_prices) < 2:  # Need at least entry and stop loss
                return None
            
            # Assume first price is entry, second is stop loss
            entry_price = float(filtered_prices[0])
            stop_loss = float(filtered_prices[1])
            
            result = {
                "symbol": symbol,
                "action": action,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "signal_date": datetime.now(self.gmt4_tz).isoformat(),
            }
            
            # Additional prices might be targets
            for i, price in enumerate(filtered_prices[2:], 1):
                result[f"target_{i}"] = float(price)
            
            return result
            
        except Exception as e:
            print(f"Error parsing natural language: {e}")
            return None
    
    def _parse_simple_format(self, text: str) -> Optional[Dict]:
        """
        Parse simple format like:
        NAS100 SELL  25482.60
        
        SL: 25592.60
        TP: 25182.60
        --Trade by Alex
        
        US30 SELL  48467.00
        
        SL: 48577.00
        TP: 48167.00
        --Trade by William
        
        Format: SYMBOL ACTION ENTRY_PRICE
        SL: STOP_LOSS
        TP: TARGET_PRICE
        """
        try:
            # Pattern: SYMBOL (can contain letters and numbers, optionally with space) followed by BUY/SELL followed by entry price
            # Match symbols like NAS100, US30, US 30, EURUSD, etc.
            # Entry price is the number immediately after BUY/SELL
            # First try with space in symbol (e.g., "US 30")
            pattern = r'\b([A-Z]{2,4})\s+(\d{1,4})\s+(BUY|SELL)\s+([\d.]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                # Symbol with space: "US 30" -> "US30"
                potential_symbol = (match.group(1) + match.group(2)).strip().upper()
                action = match.group(3).strip().lower()
                entry_price_str = match.group(4).strip()
            else:
                # Try without space: "US30" or "NAS100"
                pattern = r'\b([A-Z0-9]{3,8})\s+(BUY|SELL)\s+([\d.]+)'
                match = re.search(pattern, text, re.IGNORECASE)
                
                if not match:
                    # Fallback: Try to find US30 or other known indices anywhere, then find BUY/SELL and price nearby
                    # This handles cases where symbol might be on a different line or position
                    known_indices_pattern = r'\b(US30|NAS100|US100|US500|SPX|DJI)\b'
                    symbol_match = re.search(known_indices_pattern, text, re.IGNORECASE)
                    if symbol_match:
                        potential_symbol = symbol_match.group(1).strip().upper()
                        # Find BUY/SELL after the symbol
                        action_match = re.search(r'\b(BUY|SELL)\b', text[symbol_match.end():], re.IGNORECASE)
                        if action_match:
                            action = action_match.group(1).strip().lower()
                            # Find price after BUY/SELL
                            price_match = re.search(r'\b(\d{4,}\.?\d*)\b', text[symbol_match.end() + action_match.end():], re.IGNORECASE)
                            if price_match:
                                entry_price_str = price_match.group(1).strip()
                            else:
                                return None
                        else:
                            return None
                    else:
                        return None
                else:
                    potential_symbol = match.group(1).strip().upper()
                    action = match.group(2).strip().lower()
                    entry_price_str = match.group(3).strip()
            
            # Validate it's a trading symbol (not a common word)
            if not self._is_valid_trading_symbol(potential_symbol):
                return None
            
            try:
                symbol = self.normalize_symbol(potential_symbol)
            except ValueError:
                # Invalid symbol - skip this signal
                return None
            
            # Extract entry price
            try:
                entry_price = float(entry_price_str)
            except ValueError:
                return None
            
            # Extract Stop Loss - look for "SL:" or "SL " followed by number
            stop_loss = None
            sl_patterns = [
                r'SL[:\s]+([\d.]+)',
                r'Stop\s*Loss[:\s]+([\d.]+)',
            ]
            for pattern in sl_patterns:
                sl_match = re.search(pattern, text, re.IGNORECASE)
                if sl_match:
                    try:
                        stop_loss = float(sl_match.group(1).strip())
                        break
                    except ValueError:
                        continue
            
            # Extract Take Profit - look for "TP:" or "TP " followed by number
            # This format typically has a single TP, not TP1, TP2, etc.
            target_1 = None
            tp_patterns = [
                r'TP[:\s]+([\d.]+)',
                r'Take\s*Profit[:\s]+([\d.]+)',
            ]
            for pattern in tp_patterns:
                tp_match = re.search(pattern, text, re.IGNORECASE)
                if tp_match:
                    try:
                        target_1 = float(tp_match.group(1).strip())
                        break
                    except ValueError:
                        continue
            
            # Also check for TP1, TP2, etc. (in case there are multiple targets)
            targets = {}
            tp_matches = re.findall(r'TP(\d+)[:\s]+([\d.]+)', text, re.IGNORECASE)
            for tp_num, tp_value in tp_matches:
                try:
                    targets[f'target_{tp_num}'] = float(tp_value.strip())
                except ValueError:
                    continue
            
            # If we found TP: (without number), use it as target_1
            if target_1 and 'target_1' not in targets:
                targets['target_1'] = target_1
            
            # Sort targets by number
            sorted_targets = {}
            for i in range(1, 6):
                key = f'target_{i}'
                if key in targets:
                    sorted_targets[key] = targets[key]
            
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
            print(f"Error parsing simple format: {e}")
            return None
    
    def _parse_signal_alert_format(self, text: str) -> Optional[Dict]:
        """
        Parse SIGNAL ALERT format with emojis like:
        
        Format 1:
        SIGNAL ALERT
        BUY XAUUSD 3345.1
        • 🤑 TP1: 3346.6
        • 🤑 TP2: 3348.1
        • 🤑 TP3: 3357.1
        • 🔴 SL: 3333.1
        
        Format 2:
        Gold Sell Now Scalping @ 3350-3352 🥷
        SL : 3355 🔴
        TP1 : 3348 💥
        TP2 : 3345 💥
        
        Format 3:
        Gold Buy Now @ 3340
        SL: 3330
        TP1: 3350
        TP2: 3360
        """
        try:
            # Clean emojis and bullets for easier parsing (but keep the text)
            text_clean = re.sub(r'[•●○▪▫]', '', text)
            text_clean = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text_clean)
            
            # Extract action (BUY/SELL) - handle "Buy Now", "Sell Now", etc.
            action = None
            action_match = re.search(r'\b(BUY|SELL)\b', text_clean, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().lower()
            
            if not action:
                return None
            
            # Extract symbol
            symbol = None
            text_upper = text_clean.upper()
            
            # Check for metal names FIRST (Gold -> XAUUSD, Silver -> XAGUSD)
            # This takes priority over pattern matching to avoid false matches
            if 'GOLD' in text_upper:
                symbol = 'C:XAUUSD'
            elif 'SILVER' in text_upper:
                symbol = 'C:XAGUSD'
            
            # If no metal name found, try explicit symbols (XAUUSD, EURUSD, etc.)
            if not symbol:
                symbol_match = re.search(r'\b([A-Z0-9]{6,8})\b', text_clean)
                if symbol_match:
                    potential_symbol = symbol_match.group(1).strip().upper()
                    if self._is_valid_trading_symbol(potential_symbol):
                        try:
                            symbol = self.normalize_symbol(potential_symbol)
                        except ValueError:
                            pass
            
            if not symbol:
                return None
            
            # Extract entry price
            entry_price = None
            
            # Pattern 1: "BUY XAUUSD 3345.1" - action symbol price
            entry_pattern1 = rf'\b{action.upper()}\s+[A-Z0-9]+\s+([\d.]+)'
            match1 = re.search(entry_pattern1, text_clean, re.IGNORECASE)
            if match1:
                try:
                    entry_price = float(match1.group(1).strip())
                except ValueError:
                    pass
            
            # Pattern 2: "@ 3350-3352" - range format
            if not entry_price:
                range_pattern = r'@\s*([\d.]+)\s*[-–]\s*([\d.]+)'
                match2 = re.search(range_pattern, text_clean)
                if match2:
                    try:
                        val1 = float(match2.group(1).strip())
                        val2 = float(match2.group(2).strip())
                        entry_price = (val1 + val2) / 2  # Average of range
                    except ValueError:
                        pass
            
            # Pattern 3: "@ 3340" - single price after @
            if not entry_price:
                at_pattern = r'@\s*([\d.]+)'
                match3 = re.search(at_pattern, text_clean)
                if match3:
                    try:
                        entry_price = float(match3.group(1).strip())
                    except ValueError:
                        pass
            
            if not entry_price:
                return None
            
            # Extract Stop Loss
            stop_loss = None
            sl_patterns = [
                r'SL\s*[:\s]+\s*([\d.]+)',
                r'Stop\s*Loss\s*[:\s]+\s*([\d.]+)',
            ]
            for pattern in sl_patterns:
                sl_match = re.search(pattern, text_clean, re.IGNORECASE)
                if sl_match:
                    try:
                        stop_loss = float(sl_match.group(1).strip())
                        break
                    except ValueError:
                        continue
            
            if not stop_loss:
                return None
            
            # Extract Take Profit targets
            tp_matches = re.findall(r'TP\s*(\d+)\s*[:\s]+\s*([\d.]+)', text_clean, re.IGNORECASE)
            if not tp_matches:
                # Try without number: "TP: 3350" (assume TP1)
                tp_simple = re.search(r'TP\s*[:\s]+\s*([\d.]+)', text_clean, re.IGNORECASE)
                if tp_simple:
                    tp_matches = [('1', tp_simple.group(1))]
            
            targets = {}
            for tp_num, tp_value in tp_matches:
                try:
                    targets[f'target_{tp_num}'] = float(tp_value.strip())
                except ValueError:
                    continue
            
            # Sort targets by number
            sorted_targets = {}
            for i in range(1, 6):
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
            print(f"Error parsing signal alert format: {e}")
            return None
    
    def _parse_inline_format(self, text: str) -> Optional[Dict]:
        """
        Parse inline format like:
        Format 1:
        xauusd sell 4593/4596
        tp1 4589 
        tp2  4586
        tp3  4582
        tp4 4554
        SL 4606
        
        Format 2:
        XAUUSD BUY 4595 92 Pr
        TP: 4600
        TP2 4604
        TP3 4608
        STOP LOSS ....4590
        """
        try:
            # Pattern: SYMBOL ACTION ENTRY (where entry can be range X/Y or space-separated)
            # Match: XAUUSD BUY 4595 or xauusd sell 4593/4596 or NAS100 SELL 25482.60
            # Updated to allow numbers in symbols (for indices like NAS100, US30)
            pattern = r'\b([A-Z0-9]{3,8}|[a-z0-9]{3,8})\s+(BUY|SELL)\s+([\d./\s]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if not match:
                return None
            
            potential_symbol = match.group(1).strip().upper()
            action = match.group(2).strip().lower()
            entry_str = match.group(3).strip()
            
            # Validate it's a trading symbol (not a common word)
            if not self._is_valid_trading_symbol(potential_symbol):
                return None
            
            try:
                symbol = self.normalize_symbol(potential_symbol)
            except ValueError:
                # Invalid symbol - skip this signal
                return None
            
            # Extract entry price - can be single value, range (X/Y), or space-separated
            entry_price = None
            
            # Clean entry_str - remove non-numeric characters except /, ., and spaces
            entry_str = re.sub(r'[^\d./\s]', '', entry_str).strip()
            
            # Handle range format: 4593/4596
            if '/' in entry_str:
                parts = entry_str.split('/')
                try:
                    val1 = float(parts[0].strip())
                    val2 = float(parts[1].strip())
                    entry_price = (val1 + val2) / 2  # Average of range
                except (ValueError, IndexError):
                    pass
            # Handle space-separated: 4595 92 (could be 4595.92)
            elif ' ' in entry_str:
                parts = entry_str.split()
                try:
                    if len(parts) >= 2:
                        # Try combining: 4595 + 92 = 4595.92
                        entry_price = float(f"{parts[0]}.{parts[1]}")
                    else:
                        entry_price = float(parts[0])
                except (ValueError, IndexError):
                    pass
            else:
                try:
                    entry_price = float(entry_str)
                except ValueError:
                    pass
            
            if not entry_price:
                return None
            
            # Extract Stop Loss
            stop_loss = None
            sl_patterns = [
                r'SL[:\s]+([\d.]+)',
                r'sl\s+([\d.]+)',  # lowercase "sl 4606"
                r'STOP\s*LOSS[:\s]+([\d.]+)',
                r'STOP\s*LOSS[.\s]+([\d.]+)',  # For "STOP LOSS ....4590"
            ]
            for pattern in sl_patterns:
                sl_match = re.search(pattern, text, re.IGNORECASE)
                if sl_match:
                    try:
                        stop_loss = float(sl_match.group(1).strip())
                        break
                    except ValueError:
                        continue
            
            # Extract Take Profit targets
            targets = {}
            
            # Pattern 1: TP1, TP2, TP3, TP4 (with or without colon, handles integers and decimals)
            tp_matches = re.findall(r'TP(\d+)[:\s]+([\d.]+)', text, re.IGNORECASE)
            for tp_num, tp_value in tp_matches:
                try:
                    targets[f'target_{tp_num}'] = float(tp_value.strip())
                except ValueError:
                    continue
            
            # Pattern 2: tp1, tp2 (lowercase, no colon, handles integers)
            if not targets:
                tp_matches = re.findall(r'tp(\d+)\s+([\d.]+)', text, re.IGNORECASE)
                for tp_num, tp_value in tp_matches:
                    try:
                        targets[f'target_{tp_num}'] = float(tp_value.strip())
                    except ValueError:
                        continue
            
            # Pattern 3: TP: (without number, assume TP1)
            if not targets:
                tp_match = re.search(r'TP[:\s]+([\d.]+)', text, re.IGNORECASE)
                if tp_match:
                    try:
                        targets['target_1'] = float(tp_match.group(1).strip())
                    except ValueError:
                        pass
            
            # Sort targets by number
            sorted_targets = {}
            for i in range(1, 6):
                key = f'target_{i}'
                if key in targets:
                    sorted_targets[key] = targets[key]
            
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
            print(f"Error parsing inline format: {e}")
            return None
    
    def _parse_range_format(self, text: str) -> Optional[Dict]:
        """
        Parse range format like:
        Buy BTCUSD at any price between 94900 till 94500
        Target 1: 96000
        Target 2: 97400
        Target 3: 99000
        Target 4: 100500
        Target 5: 102000
        Stop Loss: 93500
        
        Also handles:
        📈Forex Signal
        Buy EURUSD at any price between 1.1695 till 1.1670
        Target 1: 1.1742
        Target 2: 1.1820
        Target 3: 1.1908
        Target 4: 1.2030
        Target 5: 1.2200
        Stop Loss: 1.1607
        """
        try:
            text_upper = text.upper()
            
            # Look for "at any price between X till Y" or "between X and Y" or "between X until Y" or "between X till Y"
            # Enhanced pattern to handle "at any price between" prefix
            range_match = re.search(
                r'(?:at\s+any\s+price\s+)?between\s+([\d.]+)\s+(?:till|until|and|to)\s+([\d.]+)',
                text,
                re.IGNORECASE
            )
            if not range_match:
                return None
            
            # Extract action (Buy/Sell) - should be before "at any price"
            action_match = re.search(r'\b(BUY|SELL)\b', text[:range_match.start()], re.IGNORECASE)
            if not action_match:
                return None
            
            action = action_match.group(1).strip().lower()
            
            # Extract symbol - should be before "at any price"
            # Look for known trading symbol patterns (currency pairs, crypto)
            symbol_match = None
            
            # Try currency/crypto pair pattern first (6-7 chars like EURUSD, BTCUSD) or indices with numbers (NAS100, US30)
            symbol_match = re.search(r'\b([A-Z0-9]{3,8})\b', text[:range_match.start()], re.IGNORECASE)
            if symbol_match:
                potential = symbol_match.group(1).strip().upper()
                # Check against blacklist
                blacklist = {'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 'DIRECTION', 'BETWEEN', 'TILL', 'ANY', 'FOLLOW', 'RULES', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'TP', 'SL'}
                if potential in blacklist:
                    symbol_match = None
                elif not self._is_valid_trading_symbol(potential):
                    symbol_match = None
            
            # If not found, try 3-8 char pattern but validate
            if not symbol_match:
                symbol_match = re.search(r'\b([A-Z]{3,8})\b', text[:range_match.start()], re.IGNORECASE)
                if symbol_match:
                    potential = symbol_match.group(1).strip().upper()
                    blacklist = {'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 'DIRECTION', 'BETWEEN', 'TILL', 'ANY', 'FOLLOW', 'RULES', 'CRYPTO', 'FOREX', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'TP', 'SL'}
                    if potential in blacklist:
                        return None
            
            if not symbol_match:
                return None
            
            try:
                symbol = self.normalize_symbol(symbol_match.group(1).strip().upper())
            except ValueError:
                # Invalid symbol - skip this signal
                return None
            
            # Calculate entry price - use first value from range (e.g., "between 1.1695 till 1.1670" -> entry = 1.1695)
            try:
                val1 = float(range_match.group(1).strip())
                val2 = float(range_match.group(2).strip())
                # Use first value as entry price (for BUY: higher value, for SELL: lower value)
                # Since "between X till Y" typically means X is the entry point
                entry_price = val1
            except ValueError:
                return None
            
            # Extract Targets FIRST (Target 1:, Target 2:, etc.) - Extract before stop loss
            # This ensures we capture all targets correctly, even if stop loss comes after them in the message
            # Pattern must handle targets on separate lines with blank lines in between
            targets = {}
            # Pattern: "Target 1: 113.40" - require space between Target and number, colon after number
            # Use word boundary to ensure we match the full phrase
            target_matches = re.findall(r'\bTarget\s+(\d+)\s*:\s*([\d.]+)', text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for target_num, target_value in target_matches:
                try:
                    target_num_int = int(target_num.strip())
                    target_value_float = float(target_value.strip())
                    # Validate: target should not be one of the range values
                    tolerance = 0.01
                    if abs(target_value_float - val1) < tolerance or abs(target_value_float - val2) < tolerance:
                        # Skip this target if it matches a range value
                        continue
                    # Store by target number (supports up to target 10)
                    targets[target_num_int] = target_value_float
                except (ValueError, IndexError):
                    continue
            
            # Extract Stop Loss AFTER targets - must be explicitly after "Stop Loss:" and not from the range
            # This handles cases where stop loss comes after all targets in the message (common format)
            # Try multiple patterns to handle various formats
            stop_loss = None
            
            # First, try to find "Stop Loss" in the text to see what format it's in
            sl_text_snippet = None
            sl_match_found = re.search(r'Stop\s+Loss[:\s]+([\d.]+)', text, re.IGNORECASE | re.MULTILINE)
            if sl_match_found:
                # Get context around the match for debugging
                start = max(0, sl_match_found.start() - 20)
                end = min(len(text), sl_match_found.end() + 20)
                sl_text_snippet = text[start:end]
                print(f"DEBUG: Found 'Stop Loss' in text: '{sl_text_snippet}'")
            
            # Pattern 1: "Stop Loss: 114.70" with colon (most common) - try most specific first
            sl_patterns = [
                r'Stop\s+Loss\s*:\s*([\d.]+)',     # "Stop Loss: 114.70" - most common
                r'Stop\s*Loss\s*:\s*([\d.]+)',     # "StopLoss: 114.70" - no space
                r'Stop\s+Loss[:\s]+\s*([\d.]+)',   # More flexible whitespace
                r'\bStop\s+Loss\s*:\s*([\d.]+)',   # With word boundary
            ]
            
            for idx, pattern in enumerate(sl_patterns):
                sl_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if sl_match:
                    try:
                        sl_value = float(sl_match.group(1).strip())
                        print(f"DEBUG: Pattern {idx} matched stop loss value: {sl_value}")
                        # Validate: stop loss should not be one of the range values
                        # Use a very small tolerance (0.0001) to only reject if stop loss is essentially the same as range value
                        # This prevents false rejections when stop loss is close but different from range values
                        tolerance = 0.0001  # Very small tolerance - only reject if essentially identical
                        if abs(sl_value - val1) > tolerance and abs(sl_value - val2) > tolerance:
                            # Also validate: stop loss should not match any target value
                            is_target_value = False
                            for target_val in targets.values():
                                if abs(sl_value - target_val) < 0.01:
                                    is_target_value = True
                                    print(f"DEBUG: Stop loss {sl_value} rejected (matches target value {target_val})")
                                    break
                            if not is_target_value:
                                stop_loss = sl_value
                                print(f"DEBUG: Stop loss {sl_value} validated (val1={val1}, val2={val2}, targets={list(targets.values())})")
                                break  # Found valid stop loss, stop searching
                        else:
                            print(f"DEBUG: Stop loss {sl_value} rejected (too close to range: val1={val1}, val2={val2}, diff1={abs(sl_value - val1):.6f}, diff2={abs(sl_value - val2):.6f})")
                    except ValueError as e:
                        print(f"DEBUG: Error converting stop loss value: {e}")
                        continue
            
            # If still not found, try pattern without colon
            if stop_loss is None:
                sl_match2 = re.search(r'Stop\s+Loss\s+([\d.]+)', text, re.IGNORECASE | re.MULTILINE)
                if sl_match2:
                    try:
                        sl_value = float(sl_match2.group(1).strip())
                        print(f"DEBUG: Pattern without colon matched: {sl_value}")
                        tolerance = 0.0001  # Very small tolerance - only reject if essentially identical
                        if abs(sl_value - val1) > tolerance and abs(sl_value - val2) > tolerance:
                            # Also validate: stop loss should not match any target value
                            is_target_value = False
                            for target_val in targets.values():
                                if abs(sl_value - target_val) < 0.01:
                                    is_target_value = True
                                    print(f"DEBUG: Stop loss {sl_value} rejected (matches target value {target_val})")
                                    break
                            if not is_target_value:
                                stop_loss = sl_value
                                print(f"DEBUG: Stop loss {sl_value} validated")
                        else:
                            print(f"DEBUG: Stop loss {sl_value} rejected (too close to range: val1={val1}, val2={val2})")
                    except ValueError:
                        pass
            
            if stop_loss is None:
                print(f"DEBUG: No valid stop loss found. val1={val1}, val2={val2}, targets={list(targets.values())}")
            
            # CRITICAL: Ensure stop_loss is NOT val2 or val1 (safety check)
            # If stop_loss matches val2 or val1, it means we failed to extract the real stop loss
            # Use very small tolerance (0.0001) to only reject if essentially identical
            if stop_loss is not None:
                tolerance = 0.0001  # Very small tolerance - only reject if essentially identical
                if abs(stop_loss - val2) < tolerance:
                    # Stop loss matches val2 - this is wrong, clear it
                    print(f"WARNING: Stop loss {stop_loss} matches range val2 {val2} (diff={abs(stop_loss - val2):.6f}), clearing stop_loss")
                    stop_loss = None
                elif abs(stop_loss - val1) < tolerance:
                    # Stop loss matches val1 - this is wrong, clear it
                    print(f"WARNING: Stop loss {stop_loss} matches range val1 {val1} (diff={abs(stop_loss - val1):.6f}), clearing stop_loss")
                    stop_loss = None
            
            # Build result dictionary
            result = {
                "symbol": symbol,
                "action": action,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "signal_date": datetime.now(self.gmt4_tz).isoformat(),
            }
            
            # Assign targets by their actual number (not sequentially)
            # If message has "Target 2", it should be stored as target_2, not target_1
            for target_num in sorted(targets.keys()):
                if target_num <= 5:  # Support up to target 5
                    result[f"target_{target_num}"] = targets[target_num]
            
            print(f"DEBUG _parse_range_format: Final result - entry={entry_price}, sl={stop_loss}, targets={targets}, val1={val1}, val2={val2}")
            # Final validation: if stop_loss is None but we have targets, that's suspicious
            if stop_loss is None and targets:
                print(f"WARNING: Stop loss is None but targets were found. This might indicate a parsing issue.")
            
            return result
            
        except Exception as e:
            print(f"Error parsing range format: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_emoji_format(self, text: str) -> Optional[Dict]:
        """
        Parse emoji format like:
        XAUUSD BUY_ 4595 _ 92 📈📈
        🟢 TP¹ 4600
        🟢 TP² 4604
        🟢 TP³ 4608__ OPNE
        🔴 STOP LOSS ……4590 📉
        Risk 0.1% 👤
        """
        try:
            # Pattern: SYMBOL ACTION_ ENTRY _ DECIMAL (optional)
            # Example: XAUUSD BUY_ 4595 _ 92 or NAS100 SELL_ 25482.60
            # Also handles: XAUUSD BUY_ 4403..4404 (range format) or XAUUSD BUY 4463-4460 (dash range)
            # Updated to allow numbers in symbols (for indices like NAS100, US30)
            pattern = r'\b([A-Z0-9]{3,8}|[a-z0-9]{3,8})\s+(BUY|SELL)[_\s]+([\d.]+(?:\.{2,}\d+|-\d+)?)(?:\s*[_\s]+\s*([\d.]+))?'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if not match:
                return None
            
            potential_symbol = match.group(1).strip().upper()
            action = match.group(2).strip().lower()
            entry_main = match.group(3).strip()
            entry_decimal = match.group(4).strip() if match.group(4) else None
            
            # Validate symbol
            if not self._is_valid_trading_symbol(potential_symbol):
                return None
            
            try:
                symbol = self.normalize_symbol(potential_symbol)
            except ValueError:
                return None
            
            # Parse entry price - handle ranges, multiple dots, and decimal parts
            entry_price = None
            
            # Clean up entry_main - remove trailing dots
            entry_main = entry_main.rstrip('.')
            
            # Check if entry_main contains a range (multiple dots like "4403..4404" or dash like "4463-4460")
            if '..' in entry_main or '...' in entry_main:
                # It's a range - extract the two numbers and calculate average
                # Handle patterns like: "4403..4404", "4249...50", "4496..95"
                range_match = re.search(r'(\d+)\.{2,}(\d+)', entry_main)
                if range_match:
                    try:
                        val1 = float(range_match.group(1))
                        val2_str = range_match.group(2)
                        val2 = float(val2_str)
                        
                        # Determine if it's a range or decimal part
                        # If val2 is much smaller than val1 (like 95 vs 4496), it's likely a decimal part
                        # If val2 is similar in magnitude to val1 (like 4404 vs 4403), it's likely a range
                        if len(val2_str) <= 2 and val1 > 1000 and val2 < 100:
                            # Likely a decimal part: "4496..95" = 4496.95, "4249...50" = 4249.50
                            entry_price = float(f"{val1}.{val2_str}")
                        elif abs(val1 - val2) < val1 * 0.1:  # Values are within 10% of each other
                            # Likely a range: "4403..4404" = average
                            entry_price = (val1 + val2) / 2
                        else:
                            # Default: treat as decimal if val2 is small, otherwise range
                            if val2 < 100:
                                entry_price = float(f"{val1}.{val2_str}")
                            else:
                                entry_price = (val1 + val2) / 2
                    except (ValueError, IndexError):
                        pass
            elif '-' in entry_main:
                # Handle dash-separated range like "4463-4460"
                dash_range_match = re.search(r'(\d+)\s*-\s*(\d+)', entry_main)
                if dash_range_match:
                    try:
                        val1 = float(dash_range_match.group(1))
                        val2 = float(dash_range_match.group(2))
                        # For entry ranges, use the first (higher) value or average
                        # Typically for BUY: higher value, for SELL: lower value
                        # For simplicity, use average
                        entry_price = (val1 + val2) / 2
                    except (ValueError, IndexError):
                        pass
            
            # If not a range, try to parse as regular number
            if entry_price is None:
                if entry_decimal:
                    # Combine: 4595 + 92 = 4595.92
                    try:
                        entry_price = float(f"{entry_main}.{entry_decimal}")
                    except ValueError:
                        # If that fails, try parsing entry_main alone
                        try:
                            entry_price = float(entry_main)
                        except ValueError:
                            pass
                else:
                    # Just parse entry_main
                    try:
                        entry_price = float(entry_main)
                    except ValueError:
                        pass
            
            if entry_price is None:
                return None
            
            # Extract Stop Loss - look for "STOP LOSS" followed by dots, underscores, spaces and number
            stop_loss = None
            sl_patterns = [
                r'STOP\s*LOSS[._\s]+([\d.]+)',  # STOP LOSS ____ 4454 or STOP LOSS ……4590
                r'STOP\s*LOSS[:\s]+([\d.]+)',  # STOP LOSS: 4590
                r'SL[:\s]+([\d.]+)',  # SL: 4590
            ]
            for pattern in sl_patterns:
                sl_match = re.search(pattern, text, re.IGNORECASE)
                if sl_match:
                    try:
                        sl_value = sl_match.group(1).strip().rstrip('.')  # Remove trailing dots
                        
                        # Handle ranges in stop loss (like "4590..4588")
                        if '..' in sl_value or '...' in sl_value:
                            range_match = re.search(r'(\d+)\.{2,}(\d+)', sl_value)
                            if range_match:
                                try:
                                    val1 = float(range_match.group(1))
                                    val2_str = range_match.group(2)
                                    if len(val2_str) <= 2 and val1 > 1000:
                                        # Likely decimal part
                                        stop_loss = float(f"{val1}.{val2_str}")
                                    else:
                                        # Range - use average
                                        val2 = float(val2_str)
                                        stop_loss = (val1 + val2) / 2
                                except (ValueError, IndexError):
                                    pass
                        else:
                            stop_loss = float(sl_value)
                        
                        if stop_loss is not None:
                            break
                    except ValueError:
                        continue
            
            # Extract Take Profit targets - look for TP¹, TP², TP³, etc.
            targets = {}
            
            # Pattern 1: TP¹, TP², TP³ (with superscript numbers)
            tp_superscript_patterns = [
                r'TP[¹1][:\s]+([\d.]+)',
                r'TP[²2][:\s]+([\d.]+)',
                r'TP[³3][:\s]+([\d.]+)',
                r'TP[⁴4][:\s]+([\d.]+)',
                r'TP[⁵5][:\s]+([\d.]+)',
            ]
            for i, pattern in enumerate(tp_superscript_patterns, 1):
                tp_match = re.search(pattern, text, re.IGNORECASE)
                if tp_match:
                    try:
                        targets[f'target_{i}'] = float(tp_match.group(1).strip())
                    except ValueError:
                        continue
            
            # Pattern 2: TP1, TP2, TP3 (regular numbers) - if superscript didn't match
            if not targets:
                tp_matches = re.findall(r'TP(\d+)[:\s]+([\d.]+)', text, re.IGNORECASE)
                for tp_num, tp_value in tp_matches:
                    try:
                        targets[f'target_{tp_num}'] = float(tp_value.strip())
                    except ValueError:
                        continue
            
            # Sort targets
            sorted_targets = {}
            for i in range(1, 6):
                key = f'target_{i}'
                if key in targets:
                    sorted_targets[key] = targets[key]
            
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
            print(f"Error parsing emoji format: {e}")
            return None
    
    def _parse_range_with_explanation(self, text: str) -> Optional[Dict]:
        """
        Parse range format with explanation text like:
        Buy EURUSD at any price between 1.1240 until 1.1220 = You can place 3 trades. 
        
        1st trade at 1.1240, 2nd trade at 1.1230, 3rd trade at 1.1220
        
        Take profit Target 1: 1.1290
        Target 2: 1.1380
        Target 3: 1.1460
        Stop Loss: 1.1180
        """
        try:
            # Look for "at any price between X until Y" or "between X and Y"
            range_match = re.search(
                r'at\s+any\s+price\s+between\s+([\d.]+)\s+until\s+([\d.]+)',
                text,
                re.IGNORECASE
            )
            if not range_match:
                # Try without "at any price"
                range_match = re.search(
                    r'between\s+([\d.]+)\s+until\s+([\d.]+)',
                    text,
                    re.IGNORECASE
                )
            
            if not range_match:
                return None
            
            # Extract action (Buy/Sell) - should be before "at any price" or "between"
            action_match = re.search(r'\b(BUY|SELL)\b', text[:range_match.start()], re.IGNORECASE)
            if not action_match:
                return None
            
            action = action_match.group(1).strip().lower()
            
            # Extract symbol - should be before "at any price" or "between"
            symbol_match = None
            
            # Try currency/crypto pair pattern first (6-7 chars like EURUSD, BTCUSD) or indices with numbers (NAS100, US30)
            symbol_match = re.search(r'\b([A-Z0-9]{3,8})\b', text[:range_match.start()], re.IGNORECASE)
            if symbol_match:
                potential = symbol_match.group(1).strip().upper()
                # Check against blacklist
                blacklist = {'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 'DIRECTION', 'BETWEEN', 'TILL', 'ANY', 'FOLLOW', 'RULES', 'CRYPTO', 'FOREX', 'PLACE', 'TRADES', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'TP', 'SL'}
                if potential in blacklist:
                    symbol_match = None
                elif not self._is_valid_trading_symbol(potential):
                    symbol_match = None
            
            if not symbol_match:
                return None
            
            try:
                symbol = self.normalize_symbol(symbol_match.group(1).strip().upper())
            except ValueError:
                return None
            
            # Calculate entry price - use first value from range (e.g., "between 1.1695 till 1.1670" -> entry = 1.1695)
            try:
                val1 = float(range_match.group(1).strip())
                val2 = float(range_match.group(2).strip())
                # Use first value as entry price (for BUY: higher value, for SELL: lower value)
                # Since "between X till Y" typically means X is the entry point
                entry_price = val1
            except ValueError:
                return None
            
            # Extract Targets FIRST (Target 1:, Target 2:, etc.) - Extract before stop loss
            # This ensures we capture all targets correctly, even if stop loss comes after them in the message
            # Pattern must handle targets on separate lines with blank lines in between
            targets = {}
            # Pattern: "Target 1: 113.40" - require space between Target and number, colon after number
            # Use word boundary to ensure we match the full phrase
            target_matches = re.findall(r'\bTarget\s+(\d+)\s*:\s*([\d.]+)', text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for target_num, target_value in target_matches:
                try:
                    target_num_int = int(target_num.strip())
                    target_value_float = float(target_value.strip())
                    # Validate: target should not be one of the range values
                    tolerance = 0.01
                    if abs(target_value_float - val1) < tolerance or abs(target_value_float - val2) < tolerance:
                        # Skip this target if it matches a range value
                        continue
                    # Store by target number (supports up to target 10)
                    targets[target_num_int] = target_value_float
                except (ValueError, IndexError):
                    continue
            
            # Extract Stop Loss AFTER targets - must be explicitly after "Stop Loss:" and not from the range
            # This handles cases where stop loss comes after all targets in the message (common format)
            # Try multiple patterns to handle various formats
            stop_loss = None
            
            # First, try to find "Stop Loss" in the text to see what format it's in
            sl_text_snippet = None
            sl_match_found = re.search(r'Stop\s+Loss[:\s]+([\d.]+)', text, re.IGNORECASE | re.MULTILINE)
            if sl_match_found:
                # Get context around the match for debugging
                start = max(0, sl_match_found.start() - 20)
                end = min(len(text), sl_match_found.end() + 20)
                sl_text_snippet = text[start:end]
                print(f"DEBUG: Found 'Stop Loss' in text: '{sl_text_snippet}'")
            
            # Pattern 1: "Stop Loss: 114.70" with colon (most common) - try most specific first
            sl_patterns = [
                r'Stop\s+Loss\s*:\s*([\d.]+)',     # "Stop Loss: 114.70" - most common
                r'Stop\s*Loss\s*:\s*([\d.]+)',     # "StopLoss: 114.70" - no space
                r'Stop\s+Loss[:\s]+\s*([\d.]+)',   # More flexible whitespace
                r'\bStop\s+Loss\s*:\s*([\d.]+)',   # With word boundary
            ]
            
            for idx, pattern in enumerate(sl_patterns):
                sl_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if sl_match:
                    try:
                        sl_value = float(sl_match.group(1).strip())
                        print(f"DEBUG: Pattern {idx} matched stop loss value: {sl_value}")
                        # Validate: stop loss should not be one of the range values
                        # Use a very small tolerance (0.0001) to only reject if stop loss is essentially the same as range value
                        # This prevents false rejections when stop loss is close but different from range values
                        tolerance = 0.0001  # Very small tolerance - only reject if essentially identical
                        if abs(sl_value - val1) > tolerance and abs(sl_value - val2) > tolerance:
                            # Also validate: stop loss should not match any target value
                            is_target_value = False
                            for target_val in targets.values():
                                if abs(sl_value - target_val) < 0.01:
                                    is_target_value = True
                                    print(f"DEBUG: Stop loss {sl_value} rejected (matches target value {target_val})")
                                    break
                            if not is_target_value:
                                stop_loss = sl_value
                                print(f"DEBUG: Stop loss {sl_value} validated (val1={val1}, val2={val2}, targets={list(targets.values())})")
                                break  # Found valid stop loss, stop searching
                        else:
                            print(f"DEBUG: Stop loss {sl_value} rejected (too close to range: val1={val1}, val2={val2}, diff1={abs(sl_value - val1):.6f}, diff2={abs(sl_value - val2):.6f})")
                    except ValueError as e:
                        print(f"DEBUG: Error converting stop loss value: {e}")
                        continue
            
            # If still not found, try pattern without colon
            if stop_loss is None:
                sl_match2 = re.search(r'Stop\s+Loss\s+([\d.]+)', text, re.IGNORECASE | re.MULTILINE)
                if sl_match2:
                    try:
                        sl_value = float(sl_match2.group(1).strip())
                        print(f"DEBUG: Pattern without colon matched: {sl_value}")
                        tolerance = 0.0001  # Very small tolerance - only reject if essentially identical
                        if abs(sl_value - val1) > tolerance and abs(sl_value - val2) > tolerance:
                            # Also validate: stop loss should not match any target value
                            is_target_value = False
                            for target_val in targets.values():
                                if abs(sl_value - target_val) < 0.01:
                                    is_target_value = True
                                    print(f"DEBUG: Stop loss {sl_value} rejected (matches target value {target_val})")
                                    break
                            if not is_target_value:
                                stop_loss = sl_value
                                print(f"DEBUG: Stop loss {sl_value} validated")
                        else:
                            print(f"DEBUG: Stop loss {sl_value} rejected (too close to range: val1={val1}, val2={val2})")
                    except ValueError:
                        pass
            
            if stop_loss is None:
                print(f"DEBUG: No valid stop loss found. val1={val1}, val2={val2}, targets={list(targets.values())}")
            
            # CRITICAL: Ensure stop_loss is NOT val2 or val1 (safety check)
            # If stop_loss matches val2 or val1, it means we failed to extract the real stop loss
            # Use very small tolerance (0.0001) to only reject if essentially identical
            if stop_loss is not None:
                tolerance = 0.0001  # Very small tolerance - only reject if essentially identical
                if abs(stop_loss - val2) < tolerance:
                    # Stop loss matches val2 - this is wrong, clear it
                    print(f"WARNING: Stop loss {stop_loss} matches range val2 {val2} (diff={abs(stop_loss - val2):.6f}), clearing stop_loss")
                    stop_loss = None
                elif abs(stop_loss - val1) < tolerance:
                    # Stop loss matches val1 - this is wrong, clear it
                    print(f"WARNING: Stop loss {stop_loss} matches range val1 {val1} (diff={abs(stop_loss - val1):.6f}), clearing stop_loss")
                    stop_loss = None
            
            # Build result dictionary
            result = {
                "symbol": symbol,
                "action": action,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "signal_date": datetime.now(self.gmt4_tz).isoformat(),
            }
            
            # Assign targets by their actual number (not sequentially)
            # If message has "Target 2", it should be stored as target_2, not target_1
            for target_num in sorted(targets.keys()):
                if target_num <= 5:  # Support up to target 5
                    result[f"target_{target_num}"] = targets[target_num]
            
            print(f"DEBUG _parse_range_with_explanation: Final result - entry={entry_price}, sl={stop_loss}, targets={targets}")
            
            return result
            
        except Exception as e:
            print(f"Error parsing range with explanation format: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _is_valid_trading_symbol(self, symbol: str) -> bool:
        """
        Check if a string is likely a valid trading symbol.
        Returns True if it matches known patterns or is in known lists.
        """
        symbol = symbol.upper().strip()
        
        # Blacklist of common words and invalid symbols
        blacklist = {
            'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 
            'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 
            'DIRECTION', 'BETWEEN', 'TILL', 'ANY', 'FOLLOW', 'RULES', 'QUIZ', 'AFTER',
            'REACH', 'PLACE', 'MOVE', 'HANDLE', 'BETTER', 'READ', 'HERE',
            'SLOWLY', 'COMBUY', 'SUPPORT', 'INVERSE', 'COMBINE', 'COMBINED', 'COMBINING',
            'SUPPORTS', 'SUPPORTED', 'INVERSED', 'INVERSING', 'SLOW', 'SLOWER',
            # Add TP1-TP5 and SL variations to blacklist (these are targets, not symbols)
            'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'TP', 'SL', 'STOPLOSS', 'STOP_LOSS'
        }
        if symbol in blacklist:
            return False
        
        # Known currency pairs (6-7 chars)
        known_currencies = {
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURCHF', 'AUDNZD', 'EURAUD',
            'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'EURNZD', 'EURCAD', 'AUDCAD',
            'AUDCHF', 'CADCHF', 'CADJPY', 'CHFJPY', 'NZDJPY', 'NZDCHF', 'NZDCAD',
            'XAUUSD', 'XAGUSD', 'XPDUSD', 'XPTUSD'  # Precious metals
        }
        if symbol in known_currencies:
            return True
        
        # Known crypto pairs
        known_crypto = {'BTCUSD', 'ETHUSD', 'BNBUSD', 'ADAUSD', 'SOLUSD', 'XRPUSD', 'DOGEUSD'}
        if symbol in known_crypto:
            return True
        
        # Known indices (including those with numbers)
        known_indices = {
            'SPX', 'DJI', 'NDX', 'RUT',  # Standard indices
            'NAS100', 'US30', 'US100', 'US500',  # Common index symbols
            'NAS', 'DOW', 'SP500', 'SPX500'  # Alternative names
        }
        if symbol in known_indices:
            return True
        
        # Pattern validation for indices with numbers (NAS100, US30, etc.)
        # Match patterns like: NAS100, US30, US100, etc. (3-6 chars, mix of letters and numbers)
        if re.match(r'^[A-Z]{2,4}\d{2,4}$', symbol):
            # Pattern like NAS100, US30, US100, etc.
            return True
        
        # Pattern validation: 6-7 chars, typically starts with currency codes
        if len(symbol) >= 6 and len(symbol) <= 7:
            # Check if it starts with known currency codes
            currency_codes = {'EUR', 'GBP', 'USD', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD', 'BTC', 'ETH', 'XAU', 'XAG'}
            if any(symbol.startswith(code) for code in currency_codes):
                return True
            # Only accept 6-7 char symbols if they match known patterns (not just any word)
            # Check if it ends with known currency codes (for pairs like EURUSD)
            if any(symbol.endswith(code) for code in currency_codes):
                return True
            # Reject unknown 6-7 char symbols that don't match currency patterns
            return False
        
        # 3-4 char symbols might be indices or single currencies
        if len(symbol) >= 3 and len(symbol) <= 4:
            indices = {'SPX', 'DJI', 'NDX', 'RUT', 'US30', 'US100', 'US500'}  # Include indices with numbers
            if symbol in indices:
                return True
            # Also check if it matches the pattern for indices with numbers (e.g., US30)
            if re.match(r'^[A-Z]{2,3}\d{1,2}$', symbol):
                return True
        
        return False
    
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
        
        # Validate symbol before normalizing - reject invalid symbols
        if not self._is_valid_trading_symbol(symbol):
            # Return None or raise error - caller should handle this
            raise ValueError(f"Invalid trading symbol: {symbol}")
        
        # Check if it's a currency pair (typically 6-7 chars: EURUSD, GBPUSD, etc.)
        if len(symbol) >= 6 and len(symbol) <= 7:
            # Common currency pairs
            currency_pairs = [
                'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
                'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURCHF', 'AUDNZD', 'EURAUD',
                'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'EURNZD', 'EURCAD', 'AUDCAD',
                'AUDCHF', 'CADCHF', 'CADJPY', 'CHFJPY', 'NZDJPY', 'NZDCHF', 'NZDCAD',
                'XAUUSD', 'XAGUSD', 'XPDUSD', 'XPTUSD'  # Precious metals
            ]
            
            if symbol in currency_pairs or symbol.startswith('XAU') or symbol.startswith('XAG'):
                return f"C:{symbol}"
        
        # Check if it's a crypto pair (BTCUSD, ETHUSD, etc.)
        crypto_pairs = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'ADAUSD', 'SOLUSD', 'XRPUSD', 'DOGEUSD']
        if symbol in crypto_pairs or (symbol.startswith('BTC') or symbol.startswith('ETH')):
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
        
        # Check if it's an index (including those with numbers)
        indices = {
            'SPX', 'DJI', 'NDX', 'RUT',  # Standard indices
            'NAS100', 'US30', 'US100', 'US500',  # Common index symbols
            'NAS', 'DOW', 'SP500', 'SPX500'  # Alternative names
        }
        if symbol in indices:
            # Normalize common index names
            if symbol == 'NAS100' or symbol == 'NAS':
                return "^NAS100"
            elif symbol == 'US30' or symbol == 'DOW':
                return "^US30"
            elif symbol == 'US100':
                return "^US100"
            elif symbol == 'US500' or symbol == 'SP500' or symbol == 'SPX500':
                return "^SPX"
            else:
                return f"^{symbol}"
        
        # Check if it's an index pattern (letters followed by numbers like NAS100, US30)
        if re.match(r'^[A-Z]{2,4}\d{2,4}$', symbol):
            # Pattern like NAS100, US30, US100, etc.
            # Normalize common patterns
            if symbol.startswith('NAS'):
                return "^NAS100"
            elif symbol.startswith('US'):
                if '30' in symbol or symbol == 'US30':
                    return "^US30"
                elif '100' in symbol or symbol == 'US100':
                    return "^US100"
                elif '500' in symbol or symbol == 'US500':
                    return "^SPX"
                else:
                    return f"^{symbol}"
            else:
                return f"^{symbol}"
        
        # Only add C: prefix if it's a valid 6-7 char currency-like symbol
        # (already validated by _is_valid_trading_symbol above)
        if len(symbol) >= 6 and len(symbol) <= 7:
            # Check if it starts or ends with known currency codes
            currency_codes = {'EUR', 'GBP', 'USD', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD', 'BTC', 'ETH', 'XAU', 'XAG'}
            if any(symbol.startswith(code) for code in currency_codes) or any(symbol.endswith(code) for code in currency_codes):
                return f"C:{symbol}"
        
        # Return as-is for stocks and other symbols (already validated)
        return symbol


