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
            re.search(r'[A-Z]{6,8}', text_upper) or  # EURUSD, BTCUSD format
            re.search(r'[a-z]{6,8}', message_text.lower()) or  # xauusd format (lowercase)
            '📣' in message_text  # Signal emoji
        )
        
        # Must have at least one price (decimal or integer number)
        has_price = bool(re.search(r'\d+\.\d{2,6}|\d{4,}', message_text))
        
        # Skip if doesn't meet minimum requirements
        if not (has_direction and has_symbol and has_price):
            return None
        
        # Try structured format parsing first (most common - PipXpert format)
        parsed = self._parse_structured_format(message_text)
        if parsed:
            return parsed
        
        # Try inline format: SYMBOL ACTION ENTRY (Format 1 & 2)
        parsed = self._parse_inline_format(message_text)
        if parsed:
            return parsed
        
        # Try range format: "at any price between X till Y" (Format 3)
        parsed = self._parse_range_format(message_text)
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
            
            # Pattern 3: Currency pairs anywhere (EUR/USD, GBPUSD, BTCUSD, etc.)
            if not symbol:
                # First, try to find known trading symbol patterns
                # Look for currency pairs with slash
                symbol_match = re.search(r'\b([A-Z]{3}/[A-Z]{3})\b', text, re.IGNORECASE)
                if symbol_match:
                    potential = symbol_match.group(1).strip().upper()
                    if self._is_valid_trading_symbol(potential.replace('/', '')):
                        symbol = potential
                
                # If not found, look for 6-7 char symbols (currency pairs without slash)
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
            
            # Pattern 3: First price-like number after symbol/direction (decimal or integer)
            if not entry_price:
                # Find all numbers (decimals or large integers)
                price_matches = re.findall(r'\b(\d+\.\d{2,6}|\d{4,})\b', text)
                if price_matches:
                    try:
                        entry_price = float(price_matches[0])
                    except (ValueError, IndexError):
                        pass
            
            if not entry_price:
                return None
            
            # Extract Stop Loss - more flexible
            stop_loss = None
            sl_patterns = [
                r'SL[:\s]+([\d.]+)',
                r'Stop\s*Loss[:\s]+([\d.]+)',
                r'Stop\s*Loss[.\s]+([\d.]+)',  # For "STOP LOSS ....4590"
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
            
            # Look for currency pairs or symbols (avoid common words)
            # Try to find known trading symbol patterns first
            symbol_match = None
            
            # Pattern 1: Currency pairs with slash (EUR/USD, GBP/AUD)
            symbol_match = re.search(r'\b([A-Z]{3}/[A-Z]{3})\b', text)
            
            # Pattern 2: Currency pairs without slash (EURUSD, GBPAUD, BTCUSD)
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
            
            symbol = self.normalize_symbol(symbol_match.group(1).strip().upper())
            
            # Look for price patterns (numbers with decimals or large integers)
            prices = re.findall(r'\b(\d+\.\d{2,6}|\d{4,})\b', text)
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
            # Match: XAUUSD BUY 4595 or xauusd sell 4593/4596
            pattern = r'\b([A-Z]{3,8}|[a-z]{3,8})\s+(BUY|SELL)\s+([\d./\s]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if not match:
                return None
            
            potential_symbol = match.group(1).strip().upper()
            action = match.group(2).strip().lower()
            entry_str = match.group(3).strip()
            
            # Validate it's a trading symbol (not a common word)
            if not self._is_valid_trading_symbol(potential_symbol):
                return None
            
            symbol = self.normalize_symbol(potential_symbol)
            
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
        Stop Loss: 93500
        """
        try:
            text_upper = text.upper()
            
            # Look for "at any price between X till Y" or "between X and Y"
            range_match = re.search(
                r'between\s+([\d.]+)\s+(?:till|and|to)\s+([\d.]+)',
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
            
            # Try currency/crypto pair pattern first (6-7 chars like EURUSD, BTCUSD)
            symbol_match = re.search(r'\b([A-Z]{6,7})\b', text[:range_match.start()], re.IGNORECASE)
            if symbol_match:
                potential = symbol_match.group(1).strip().upper()
                # Check against blacklist
                blacklist = {'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 'DIRECTION', 'BETWEEN', 'TILL', 'ANY', 'FOLLOW', 'RULES'}
                if potential in blacklist:
                    symbol_match = None
            
            # If not found, try 3-8 char pattern but validate
            if not symbol_match:
                symbol_match = re.search(r'\b([A-Z]{3,8})\b', text[:range_match.start()], re.IGNORECASE)
                if symbol_match:
                    potential = symbol_match.group(1).strip().upper()
                    blacklist = {'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 'DIRECTION', 'BETWEEN', 'TILL', 'ANY', 'FOLLOW', 'RULES', 'CRYPTO', 'FOREX'}
                    if potential in blacklist:
                        return None
            
            if not symbol_match:
                return None
            
            symbol = self.normalize_symbol(symbol_match.group(1).strip().upper())
            
            # Calculate entry price (average of range)
            try:
                val1 = float(range_match.group(1).strip())
                val2 = float(range_match.group(2).strip())
                entry_price = (val1 + val2) / 2
            except ValueError:
                return None
            
            # Extract Stop Loss
            stop_loss = None
            sl_match = re.search(r'Stop\s*Loss[:\s]+([\d.]+)', text, re.IGNORECASE)
            if sl_match:
                try:
                    stop_loss = float(sl_match.group(1).strip())
                except ValueError:
                    pass
            
            # Extract Targets (Target 1:, Target 2:, etc.)
            targets = {}
            target_matches = re.findall(r'Target\s*(\d+)[:\s]+([\d.]+)', text, re.IGNORECASE)
            for target_num, target_value in target_matches:
                try:
                    targets[f'target_{target_num}'] = float(target_value.strip())
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
            print(f"Error parsing range format: {e}")
            return None
    
    def _is_valid_trading_symbol(self, symbol: str) -> bool:
        """
        Check if a string is likely a valid trading symbol.
        Returns True if it matches known patterns or is in known lists.
        """
        symbol = symbol.upper().strip()
        
        # Blacklist of common words
        blacklist = {
            'SIGNAL', 'FOREX', 'CRYPTO', 'TRADING', 'ANALYSIS', 'TARGET', 'ENTRY', 
            'PRICE', 'STOP', 'LOSS', 'PROFIT', 'RISK', 'OPEN', 'CLOSE', 'BUY', 'SELL', 
            'DIRECTION', 'BETWEEN', 'TILL', 'ANY', 'FOLLOW', 'RULES', 'QUIZ', 'AFTER',
            'REACH', 'PLACE', 'MOVE', 'HANDLE', 'BETTER', 'READ', 'HERE'
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
        
        # Pattern validation: 6-7 chars, typically starts with currency codes
        if len(symbol) >= 6 and len(symbol) <= 7:
            # Check if it starts with known currency codes
            currency_codes = {'EUR', 'GBP', 'USD', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD', 'BTC', 'ETH', 'XAU', 'XAG'}
            if any(symbol.startswith(code) for code in currency_codes):
                return True
            # If it's 6-7 chars and not in blacklist, might be valid
            return True
        
        # 3-4 char symbols might be indices or single currencies
        if len(symbol) >= 3 and len(symbol) <= 4:
            indices = {'SPX', 'DJI', 'NDX', 'RUT'}
            if symbol in indices:
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
        
        # Check if it's an index
        indices = ['SPX', 'DJI', 'NDX', 'RUT']
        if symbol in indices:
            return f"^{symbol}"
        
        # Default: assume currency pair if 6-7 chars
        if len(symbol) >= 6 and len(symbol) <= 7:
            return f"C:{symbol}"
        
        # Return as-is for stocks and other symbols
        return symbol


