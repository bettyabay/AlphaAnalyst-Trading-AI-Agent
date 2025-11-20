"""
Enhanced Data Engine for AlphaAnalyst Trading AI Agent
- Comprehensive instrument data fetching
- AI-enhanced research document processing and indexing
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Any, BinaryIO
from datetime import datetime, timedelta
import pandas as pd

from .polygon_integration import PolygonDataClient
from .document_manager import DocumentManager
from ..database.config import get_supabase


class EnhancedDataEngine:
    def __init__(self):
        self.polygon = PolygonDataClient()
        self.docs = DocumentManager()
        self.supabase = get_supabase()

    def fetch_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch 5Y historicals, compute basic indicators, and fetch news placeholder."""
        try:
            end_date = datetime.utcnow().date()
            start_date = (datetime.utcnow() - timedelta(days=365 * 5 + 30)).date()
            df = self.polygon.get_historical_data(symbol, start_date.isoformat(), end_date.isoformat())
            if df is None or df.empty:
                return {"success": False, "symbol": symbol, "message": "No data from Polygon"}

            # Compute simple indicators
            df = df.sort_values("timestamp").set_index("timestamp")
            indicators = {}
            for window in (20, 50, 200):
                indicators[f"ma_{window}"] = df["close"].rolling(window=window).mean().iloc[-1] if len(df) >= window else None
            indicators["rsi_14"] = self._compute_rsi(df["close"], 14).iloc[-1] if len(df) >= 20 else None

            payload = {
                "success": True,
                "symbol": symbol,
                "historical": df.tail(252).reset_index().to_dict(orient="records"),  # last ~1Y for UI
                "indicators": indicators,
                "corporate_actions": [],  # Placeholder hook
            }
            return payload
        except Exception as e:
            return {"success": False, "symbol": symbol, "error": str(e)}

    def process_master_documents(self, symbol: str, uploaded_files: List[BinaryIO]) -> Dict[str, Any]:
        """Analyze uploaded research docs, categorize, sentiment score, and store AI insights.
        uploaded_files: list of file-like objects with attributes .name and file buffer
        """
        results: List[Dict[str, Any]] = []
        try:
            if not self.supabase:
                return {"success": False, "message": "Supabase not configured"}

            for f in uploaded_files:
                filename = getattr(f, "name", "uploaded_file")
                # Upload and store content
                upload_res = self.docs.upload_document(f, filename, title=filename, symbol=symbol)
                if not upload_res.get("success"):
                    results.append({"filename": filename, "success": False, "message": upload_res.get("message")})
                    continue

                doc_id = upload_res.get("document_id")
                # AI analysis + signals
                analysis = self.docs.analyze_document_with_ai(doc_id, symbol)
                signals = self.docs.extract_trading_signals(doc_id)

                # Compute sentiment score from signals
                sentiment_score = 0.0
                try:
                    if signals.get("success"):
                        sentiment_score = float(signals.get("sentiment_score", 0))
                except Exception:
                    sentiment_score = 0.0

                # Save insights into research_documents table
                try:
                    self.supabase.table("research_documents").update({
                        "ai_insights": analysis,
                        "sentiment_score": sentiment_score,
                    }).eq("id", doc_id).execute()
                except Exception:
                    # Non-fatal; continue
                    pass

                results.append({
                    "filename": filename,
                    "document_id": doc_id,
                    "success": True,
                    "signals": signals,
                    "analysis": analysis,
                    "sentiment_score": sentiment_score,
                })

            return {"success": True, "symbol": symbol, "processed": results}
        except Exception as e:
            return {"success": False, "symbol": symbol, "error": str(e)}

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def close(self):
        self.docs.close()
