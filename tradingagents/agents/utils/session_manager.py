"""
Phase 4: Session Management & Trade Execution
Manages trading sessions, active trades, and enforces concurrency limits
"""
import yfinance as yf
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ...database.config import get_supabase
from ...database.db_service import log_event, _make_json_serializable


class TradingSessionManager:
    """Manages trading sessions and tracks active trades"""
    
    MAX_CONCURRENT_TRADES = 3  # Maximum number of active trades allowed
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.supabase = get_supabase()
    
    def create_session(self, session_name: Optional[str] = None, notes: Optional[str] = None) -> Optional[Dict]:
        """Create a new trading session"""
        if not self.supabase:
            raise ValueError("Database not configured")
        
        session_data = {
            "user_id": self.user_id,
            "session_name": session_name or f"Trading Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "notes": notes or "",
            "status": "active",
            "start_date": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        try:
            # Check if 'trading_sessions' table exists, if not use system_logs
            try:
                result = self.supabase.table("trading_sessions").insert(session_data).execute()
                if result.data:
                    session = result.data[0]
                    log_event("trading_session_created", {
                        "session_id": session.get("id"),
                        "session_name": session_data["session_name"],
                        "user_id": self.user_id
                    })
                    return session
            except Exception:
                # Fallback to system_logs if table doesn't exist
                session_data["id"] = datetime.now().strftime("%Y%m%d%H%M%S")
                log_event("trading_session_created", session_data)
                return session_data
        except Exception as e:
            print(f"Error creating session: {e}")
            raise
    
    def get_active_session(self) -> Optional[Dict]:
        """Get the current active trading session"""
        if not self.supabase:
            return None
        
        try:
            # Try to get from trading_sessions table
            try:
                result = self.supabase.table("trading_sessions")\
                    .select("*")\
                    .eq("user_id", self.user_id)\
                    .eq("status", "active")\
                    .order("created_at", desc=True)\
                    .limit(1)\
                    .execute()
                
                if result.data:
                    return result.data[0]
            except Exception:
                pass
            
            # Fallback: Get from session state or return default session
            return {
                "id": "default_session",
                "session_name": "Default Trading Session",
                "status": "active",
                "start_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting active session: {e}")
            return None
    
    def close_session(self, session_id: str, notes: Optional[str] = None) -> bool:
        """Close a trading session"""
        if not self.supabase:
            return False
        
        try:
            update_data = {
                "status": "closed",
                "end_date": datetime.now().isoformat(),
                "notes": notes or ""
            }
            
            try:
                self.supabase.table("trading_sessions")\
                    .update(update_data)\
                    .eq("id", session_id)\
                    .eq("user_id", self.user_id)\
                    .execute()
            except Exception:
                # Table doesn't exist, just log
                pass
            
            log_event("trading_session_closed", {
                "session_id": session_id,
                "user_id": self.user_id,
                "notes": notes
            })
            return True
        except Exception as e:
            print(f"Error closing session: {e}")
            return False
    
    def get_all_sessions(self, limit: int = 50) -> List[Dict]:
        """Get all trading sessions for the user"""
        if not self.supabase:
            return []
        
        try:
            try:
                result = self.supabase.table("trading_sessions")\
                    .select("*")\
                    .eq("user_id", self.user_id)\
                    .order("created_at", desc=True)\
                    .limit(limit)\
                    .execute()
                
                return result.data if result.data else []
            except Exception:
                # Table doesn't exist, return empty list
                return []
        except Exception as e:
            print(f"Error getting sessions: {e}")
            return []


class TradeExecutionService:
    """Handles trade execution with concurrency limits and tracking"""
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.supabase = get_supabase()
        self.session_manager = TradingSessionManager(user_id)
    
    def get_active_trades(self) -> List[Dict]:
        """Get all active trades for the user"""
        if not self.supabase:
            return []
        
        try:
            # Check portfolio table for active positions
            result = self.supabase.table("portfolio")\
                .select("*")\
                .eq("user_id", self.user_id)\
                .execute()
            
            active_trades = []
            if result.data:
                for position in result.data:
                    # Check if there's a corresponding trade record
                    trade_record = self._get_trade_record(position.get("symbol"))
                    if trade_record or position.get("quantity", 0) > 0:
                        # Get current price and calculate P&L
                        current_price = self._get_current_price(position.get("symbol"))
                        avg_price = position.get("avg_price", 0)
                        quantity = position.get("quantity", 0)
                        unrealized_pnl = (current_price - avg_price) * quantity if current_price else position.get("pnl", 0)
                        
                        trade_data = {
                            "id": trade_record.get("id") if trade_record else position.get("id"),
                            "symbol": position.get("symbol"),
                            "quantity": quantity,
                            "entry_price": avg_price,
                            "current_price": current_price,
                            "entry_date": trade_record.get("entry_date") if trade_record else position.get("updated_at"),
                            "unrealized_pnl": unrealized_pnl,
                            "unrealized_pnl_pct": ((current_price - avg_price) / avg_price * 100) if avg_price > 0 and current_price else 0,
                            "trade_type": trade_record.get("trade_type", "LONG") if trade_record else "LONG",
                            "status": "active",
                            "session_id": trade_record.get("session_id") if trade_record else None,
                            "stop_loss": trade_record.get("stop_loss") if trade_record else None,
                            "target1": trade_record.get("target1") if trade_record else None,
                            "target2": trade_record.get("target2") if trade_record else None
                        }
                        active_trades.append(trade_data)
            
            return active_trades
        except Exception as e:
            print(f"Error getting active trades: {e}")
            return []
    
    def can_open_trade(self) -> tuple:
        """Check if we can open a new trade (enforce 3-trade limit)"""
        active_trades = self.get_active_trades()
        active_count = len(active_trades)
        
        if active_count >= TradingSessionManager.MAX_CONCURRENT_TRADES:
            return False, f"Maximum {TradingSessionManager.MAX_CONCURRENT_TRADES} concurrent trades allowed. Currently have {active_count} active trades."
        
        return True, f"Can open trade. Current active trades: {active_count}/{TradingSessionManager.MAX_CONCURRENT_TRADES}"
    
    def open_trade(
        self,
        symbol: str,
        quantity: float,
        entry_price: Optional[float] = None,
        trade_type: str = "LONG",
        stop_loss: Optional[float] = None,
        target1: Optional[float] = None,
        target2: Optional[float] = None,
        notes: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Open a new trade with concurrency check"""
        # Check if we can open a trade
        can_open, message = self.can_open_trade()
        if not can_open:
            raise ValueError(message)
        
        # Check if trade already exists for this symbol
        existing_trades = [t for t in self.get_active_trades() if t["symbol"] == symbol.upper()]
        if existing_trades:
            raise ValueError(f"Active trade already exists for {symbol}. Close existing trade first.")
        
        # Get current price if not provided
        if entry_price is None:
            entry_price = self._get_current_price(symbol)
            if entry_price is None:
                raise ValueError(f"Could not fetch current price for {symbol}")
        
        # Get or create active session
        if not session_id:
            session = self.session_manager.get_active_session()
            session_id = session.get("id") if session else None
        
        # Create trade record
        trade_data = {
            "user_id": self.user_id,
            "symbol": symbol.upper(),
            "quantity": float(quantity),
            "entry_price": float(entry_price),
            "trade_type": trade_type.upper(),
            "entry_date": datetime.now().isoformat(),
            "status": "active",
            "stop_loss": float(stop_loss) if stop_loss else None,
            "target1": float(target1) if target1 else None,
            "target2": float(target2) if target2 else None,
            "session_id": session_id,
            "notes": notes or "",
            "created_at": datetime.now().isoformat()
        }
        
        try:
            # Try to insert into trades table
            try:
                result = self.supabase.table("trades").insert(_make_json_serializable(trade_data)).execute()
                if result.data:
                    trade = result.data[0]
                else:
                    trade = trade_data
            except Exception:
                # Table doesn't exist, use trade_data as is
                trade = trade_data
            
            # Update portfolio
            self._update_portfolio(symbol, quantity, entry_price, operation="add")
            
            # Log trade execution
            log_event("trade_opened", {
                "trade_id": trade.get("id", "unknown"),
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "trade_type": trade_type,
                "session_id": session_id,
                "user_id": self.user_id
            })
            
            return trade
        except Exception as e:
            print(f"Error opening trade: {e}")
            raise
    
    def close_trade(
        self,
        symbol: str,
        exit_price: Optional[float] = None,
        quantity: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Optional[Dict]:
        """Close an active trade"""
        # Get active trade
        active_trades = [t for t in self.get_active_trades() if t["symbol"] == symbol.upper()]
        if not active_trades:
            raise ValueError(f"No active trade found for {symbol}")
        
        trade = active_trades[0]
        
        # Get exit price if not provided
        if exit_price is None:
            exit_price = self._get_current_price(symbol)
            if exit_price is None:
                raise ValueError(f"Could not fetch current price for {symbol}")
        
        # Use provided quantity or close entire position
        close_quantity = quantity if quantity else trade["quantity"]
        if close_quantity > trade["quantity"]:
            raise ValueError(f"Cannot close {close_quantity} shares. Only {trade['quantity']} shares available.")
        
        # Calculate P&L
        entry_price = trade["entry_price"]
        realized_pnl = (exit_price - entry_price) * close_quantity
        realized_pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        # Update trade record
        close_data = {
            "status": "closed" if close_quantity == trade["quantity"] else "partial",
            "exit_price": float(exit_price),
            "exit_date": datetime.now().isoformat(),
            "realized_pnl": float(realized_pnl),
            "realized_pnl_pct": float(realized_pnl_pct),
            "close_notes": notes or ""
        }
        
        try:
            trade_id = trade.get("id")
            if trade_id:
                try:
                    self.supabase.table("trades")\
                        .update(_make_json_serializable(close_data))\
                        .eq("id", trade_id)\
                        .execute()
                except Exception:
                    pass  # Table might not exist
            
            # Update portfolio
            if close_quantity == trade["quantity"]:
                # Remove from portfolio if closing entire position
                self._update_portfolio(symbol, close_quantity, entry_price, operation="remove")
            else:
                # Update quantity if partial close
                remaining_quantity = trade["quantity"] - close_quantity
                self._update_portfolio(symbol, remaining_quantity, entry_price, operation="update")
            
            # Log trade closure
            log_event("trade_closed", {
                "trade_id": trade_id or "unknown",
                "symbol": symbol,
                "quantity": close_quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": realized_pnl_pct,
                "user_id": self.user_id
            })
            
            close_data.update({
                "symbol": symbol,
                "quantity": close_quantity,
                "entry_price": entry_price,
                "exit_price": exit_price
            })
            
            return close_data
        except Exception as e:
            print(f"Error closing trade: {e}")
            raise
    
    def update_trade_pnl(self, symbol: str) -> Optional[Dict]:
        """Update unrealized P&L for a trade based on current price"""
        active_trades = [t for t in self.get_active_trades() if t["symbol"] == symbol.upper()]
        if not active_trades:
            return None
        
        trade = active_trades[0]
        current_price = self._get_current_price(symbol)
        
        if current_price is None:
            return None
        
        entry_price = trade["entry_price"]
        quantity = trade["quantity"]
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        # Update portfolio P&L
        try:
            from ...database.db_service import update_portfolio_pnl
            update_portfolio_pnl(self.user_id, symbol, current_price)
        except Exception:
            pass
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct
        }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history for the user"""
        if not self.supabase:
            return []
        
        try:
            try:
                result = self.supabase.table("trades")\
                    .select("*")\
                    .eq("user_id", self.user_id)\
                    .order("created_at", desc=True)\
                    .limit(limit)\
                    .execute()
                
                return result.data if result.data else []
            except Exception:
                # Table doesn't exist, get from logs
                from ...database.db_service import get_logs_by_event
                logs = get_logs_by_event("trade_closed", limit=limit)
                trades = []
                for log in logs:
                    details = log.get("details", {})
                    if details:
                        trades.append(details)
                return trades
        except Exception as e:
            print(f"Error getting trade history: {e}")
            return []
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            return float(price) if price else None
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _get_trade_record(self, symbol: str) -> Optional[Dict]:
        """Get trade record from database"""
        if not self.supabase:
            return None
        
        try:
            try:
                result = self.supabase.table("trades")\
                    .select("*")\
                    .eq("user_id", self.user_id)\
                    .eq("symbol", symbol.upper())\
                    .eq("status", "active")\
                    .limit(1)\
                    .execute()
                
                return result.data[0] if result.data else None
            except Exception:
                return None
        except Exception as e:
            print(f"Error getting trade record: {e}")
            return None
    
    def _update_portfolio(self, symbol: str, quantity: float, price: float, operation: str = "add"):
        """Update portfolio table"""
        if not self.supabase:
            return
        
        try:
            from ...database.db_service import add_to_portfolio, remove_from_portfolio
            
            if operation == "remove":
                remove_from_portfolio(self.user_id, symbol)
            elif operation == "add":
                add_to_portfolio(self.user_id, symbol, quantity, price)
            elif operation == "update":
                # Get current position
                result = self.supabase.table("portfolio")\
                    .select("*")\
                    .eq("user_id", self.user_id)\
                    .eq("symbol", symbol.upper())\
                    .execute()
                
                if result.data:
                    current = result.data[0]
                    current_qty = current.get("quantity", 0)
                    current_price = current.get("avg_price", 0)
                    
                    # Calculate new average price
                    if quantity > 0:
                        new_qty = quantity
                        # Keep same avg price for partial close
                        add_to_portfolio(self.user_id, symbol, new_qty, current_price)
        except Exception as e:
            print(f"Error updating portfolio: {e}")

