"""
Trading Session Manager for AlphaAnalyst Trading AI Agent
- Multi-session per day
- Enforce max 3 concurrent active trades across sessions
- Track Phase 1/Phase 2 results and executed trades
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..database.config import get_supabase


MAX_CONCURRENT_TRADES = 3


class TradingSessionManager:
    def __init__(self):
        self.supabase = get_supabase()
        # In-memory fallback if Supabase tables are unavailable
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._active_trades: Dict[str, Dict[str, Any]] = {}

    # --- Helpers ---
    def _now(self) -> str:
        return datetime.now().isoformat()

    def _sb_safe(self) -> bool:
        return self.supabase is not None

    def _active_trades_count(self) -> int:
        if self._sb_safe():
            try:
                resp = self.supabase.table("active_trades").select("status").in_("status", ["monitoring", "tp1_hit", "tp2_hit"]).execute()
                data = resp.data or []
                # Count non-closed trades
                return len([r for r in data if r.get("status") not in ("closed", "stopped_out")])
            except Exception:
                pass
        # Fallback in-memory
        return len([t for t in self._active_trades.values() if t.get("status") not in ("closed", "stopped_out")])

    def can_execute_new_trades(self, requested: int) -> bool:
        return self._active_trades_count() + requested <= MAX_CONCURRENT_TRADES

    # --- Session lifecycle ---
    def start_session(self) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "start_time": self._now(),
            "phase1_results": None,
            "phase2_results": None,
            "executed_trades": [],
            "session_status": "active",
        }
        if self._sb_safe():
            try:
                self.supabase.table("trading_sessions").insert(session).execute()
            except Exception:
                self._sessions[session_id] = session
        else:
            self._sessions[session_id] = session
        return session

    def record_phase1_results(self, session_id: str, results: Dict[str, Any]) -> bool:
        if self._sb_safe():
            try:
                self.supabase.table("trading_sessions").update({"phase1_results": results}).eq("session_id", session_id).execute()
                return True
            except Exception:
                pass
        # Fallback
        if session_id in self._sessions:
            self._sessions[session_id]["phase1_results"] = results
            return True
        return False

    def record_phase2_results(self, session_id: str, results: Dict[str, Any]) -> bool:
        if self._sb_safe():
            try:
                self.supabase.table("trading_sessions").update({"phase2_results": results}).eq("session_id", session_id).execute()
                return True
            except Exception:
                pass
        if session_id in self._sessions:
            self._sessions[session_id]["phase2_results"] = results
            return True
        return False

    # --- Trades ---
    def execute_trades(self, session_id: str, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.can_execute_new_trades(len(trades)):
            return {"success": False, "message": f"Concurrency limit {MAX_CONCURRENT_TRADES} exceeded"}

        executed: List[Dict[str, Any]] = []
        for t in trades:
            trade_id = str(uuid.uuid4())
            record = {
                "trade_id": trade_id,
                "session_id": session_id,
                "symbol": t.get("symbol"),
                "entry_data": t,
                "current_pnl": 0.0,
                "status": "monitoring",
            }
            if self._sb_safe():
                try:
                    self.supabase.table("active_trades").insert(record).execute()
                except Exception:
                    self._active_trades[trade_id] = record
            else:
                self._active_trades[trade_id] = record
            executed.append(record)

        # Save executed list to session
        if self._sb_safe():
            try:
                self.supabase.table("trading_sessions").update({"executed_trades": executed}).eq("session_id", session_id).execute()
            except Exception:
                self._sessions.setdefault(session_id, {}).setdefault("executed_trades", []).extend(executed)
        else:
            self._sessions.setdefault(session_id, {}).setdefault("executed_trades", []).extend(executed)

        return {"success": True, "executed": executed, "active_trades_count": self._active_trades_count()}

    def update_trade_status(self, trade_id: str, status: str, current_pnl: Optional[float] = None) -> bool:
        updates: Dict[str, Any] = {"status": status}
        if current_pnl is not None:
            updates["current_pnl"] = current_pnl
        if self._sb_safe():
            try:
                self.supabase.table("active_trades").update(updates).eq("trade_id", trade_id).execute()
                return True
            except Exception:
                pass
        if trade_id in self._active_trades:
            self._active_trades[trade_id].update(updates)
            return True
        return False

    def close_trade(self, trade_id: str) -> bool:
        return self.update_trade_status(trade_id, "closed")

    # --- Queries ---
    def list_active_trades(self) -> List[Dict[str, Any]]:
        if self._sb_safe():
            try:
                resp = self.supabase.table("active_trades").select("*").execute()
                data = resp.data or []
                return [r for r in data if r.get("status") not in ("closed",)]
            except Exception:
                pass
        return [t for t in self._active_trades.values() if t.get("status") not in ("closed",)]

    def list_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if self._sb_safe():
            try:
                query = self.supabase.table("trading_sessions").select("*")
                if status:
                    query = query.eq("session_status", status)
                resp = query.execute()
                return resp.data or []
            except Exception:
                pass
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.get("session_status") == status]
        return sessions

    def complete_session(self, session_id: str) -> bool:
        if self._sb_safe():
            try:
                self.supabase.table("trading_sessions").update({"session_status": "completed"}).eq("session_id", session_id).execute()
                return True
            except Exception:
                pass
        if session_id in self._sessions:
            self._sessions[session_id]["session_status"] = "completed"
            return True
        return False
