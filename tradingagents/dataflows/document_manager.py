"""
Document management system for AlphaAnalyst Trading AI Agent
"""
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, BinaryIO

from ..database.config import get_supabase

class DocumentManager:
    """Document management for research documents"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self._ensure_upload_dir()
        self.supabase = get_supabase()
    
    def _ensure_upload_dir(self):
        """Ensure upload directory exists"""
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)
    
    def upload_document(self, 
                       file_content: BinaryIO, 
                       filename: str, 
                       title: str,
                       document_type: str = "research",
                       symbol: Optional[str] = None,
                       source: str = "user_upload") -> Dict:
        """Upload and store a document"""
        try:
            # Save locally for reference
            file_extension = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(self.upload_dir, unique_filename)
            with open(file_path, "wb") as f:
                f.write(file_content.getbuffer())

            content_text = None
            try:
                # Attempt to read text if small text file
                with open(file_path, "r", encoding="utf-8") as tf:
                    content_text = tf.read()
            except Exception:
                content_text = None

            if self.supabase:
                # Insert into Supabase documents table per actual schema
                payload = {
                    "content": content_text or f"Document: {filename}",
                    "metadata": {
                        "symbol": symbol,
                        "file_name": filename,
                        "source": "user_upload"
                    },
                    "embedding": None,
                    "user_id": None
                }
                try:
                    resp = self.supabase.table("documents").insert(payload).execute()
                    print(f"Supabase insert response: {resp}")
                except Exception as e:
                    print(f"Supabase insert error details: {e}")
                    return {"success": False, "error": str(e), "message": "Supabase insert failed"}
                return {"success": True, "message": "Document uploaded to Supabase", "file_path": file_path}
            else:
                return {"success": False, "error": "Supabase not configured", "message": "Cannot upload document"}
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to upload document"
            }
    
    def get_documents(self, symbol: Optional[str] = None, 
                     document_type: Optional[str] = None) -> List[Dict]:
        """Get documents with optional filtering"""
        try:
            if not self.supabase:
                print("Supabase not configured. Cannot retrieve documents.")
                return []
                
            query = self.supabase.table("documents").select("*")
            # Filter by symbol in metadata if symbol provided
            if symbol:
                # Note: This is a simplified filter - in practice you might need more complex filtering
                resp = query.execute()
                rows = resp.data if hasattr(resp, "data") else []
                # Filter by symbol in metadata
                filtered_rows = []
                for row in rows:
                    metadata = row.get("metadata", {})
                    if metadata.get("symbol") == symbol:
                        filtered_rows.append(row)
                return filtered_rows
            else:
                resp = query.execute()
                rows = resp.data if hasattr(resp, "data") else []
                return rows
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def get_document_content(self, document_id: str) -> Optional[str]:
        """Get document content by ID"""
        try:
            if not self.supabase:
                print("Supabase not configured. Cannot retrieve document content.")
                return None
                
            resp = self.supabase.table("documents").select("content").eq("id", document_id).execute()
            data = resp.data if hasattr(resp, "data") else []
            if data and len(data) > 0:
                return data[0].get("content")
            return None
        except Exception as e:
            print(f"Error reading document {document_id}: {e}")
            return None
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        try:
            if not self.supabase:
                print("Supabase not configured. Cannot delete document.")
                return False
                
            # Delete from Supabase
            resp = self.supabase.table("documents").delete().eq("id", document_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_document_stats(self) -> Dict:
        """Get document statistics"""
        try:
            if not self.supabase:
                print("Supabase not configured. Cannot get document stats.")
                return {"total_documents": 0, "by_type": {}, "by_source": {}}
                
            # Get all documents from Supabase
            resp = self.supabase.table("documents").select("*").execute()
            documents = resp.data if hasattr(resp, "data") else []
            
            total_docs = len(documents)
            docs_by_symbol = {}
            
            for doc in documents:
                metadata = doc.get("metadata", {})
                symbol = metadata.get("symbol", "Unknown")
                if symbol not in docs_by_symbol:
                    docs_by_symbol[symbol] = 0
                docs_by_symbol[symbol] += 1
            
            return {
                "total_documents": total_docs,
                "by_symbol": docs_by_symbol
            }
        except Exception as e:
            print(f"Error getting document stats: {e}")
            return {"total_documents": 0, "by_symbol": {}}
    
    def close(self):
        """Close database connection"""
        # No database connection to close for Supabase
        pass
