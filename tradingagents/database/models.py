"""
Supabase table schemas for AlphaAnalyst Trading AI Agent
These are reference schemas matching your Supabase tables
"""

# Supabase table schemas (for reference)
SUPABASE_TABLES = {
    "market_data": {
        "id": "bigint (auto-increment)",
        "symbol": "text",
        "date": "date", 
        "open": "numeric",
        "high": "numeric",
        "low": "numeric",
        "close": "numeric",
        "volume": "bigint",
        "source": "text",
        "created_at": "timestamp with time zone"
    },
    "documents": {
        "id": "uuid (auto-generated)",
        "symbol": "text",
        "file_name": "text",
        "file_content": "text",
        "embedding_vector": "jsonb",
        "uploaded_at": "timestamp with time zone"
    },
    "positions": {
        "id": "uuid (auto-generated)",
        "user_id": "uuid",
        "symbol": "text",
        "quantity": "numeric",
        "avg_price": "numeric", 
        "pnl": "numeric",
        "updated_at": "timestamp with time zone"
    },
    "system_logs": {
        "id": "bigint (auto-increment)",
        "event": "text",
        "details": "jsonb",
        "timestamp": "timestamp with time zone"
    },
    "signals": {
        "id": "uuid (auto-generated)",
        "symbol": "text",
        "signal_type": "text",
        "confidence": "numeric",
        "details": "jsonb",
        "timestamp": "timestamp with time zone"
    },
    "users": {
        "id": "uuid (auto-generated)",
        "email": "text",
        "role": "text",
        "created_at": "timestamp with time zone"
    },
    "data_health": {
        "symbol": "text",
        "data_fetch_status": "jsonb",
        "last_updated": "timestamp with time zone",
        "health_score": "numeric"
    }
}
