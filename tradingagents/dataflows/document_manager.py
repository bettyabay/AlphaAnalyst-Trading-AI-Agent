"""
Document management system for AlphaAnalyst Trading AI Agent
Enhanced for Phase 2 with PDF processing and AI analysis
"""
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, BinaryIO
import PyPDF2
import docx
import textract
from sentence_transformers import SentenceTransformer
import numpy as np

from ..database.config import get_supabase

class DocumentManager:
    """Document management for research documents"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self._ensure_upload_dir()
        self.supabase = get_supabase()
        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None
    
    def _ensure_upload_dir(self):
        """Ensure upload directory exists"""
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)
    
    def _extract_text_from_file(self, file_path: str, file_extension: str) -> str:
        """Extract text content from various file formats"""
        try:
            if file_extension.lower() == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_extension.lower() in ['.docx', '.doc']:
                return self._extract_docx_text(file_path)
            elif file_extension.lower() in ['.txt']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Try textract as fallback
                return textract.process(file_path).decode('utf-8')
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return f"Error extracting text from {file_path}"
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text"""
        if not self.embedding_model or not text:
            return None
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
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
            
            # Handle different file input types
            if hasattr(file_content, 'getbuffer'):
                # Streamlit file uploader
                with open(file_path, "wb") as f:
                    f.write(file_content.getbuffer())
            elif hasattr(file_content, 'read'):
                # Regular file object
                with open(file_path, "wb") as f:
                    f.write(file_content.read())
            else:
                # Assume it's bytes
                with open(file_path, "wb") as f:
                    f.write(file_content)

            # Extract text content from the uploaded file
            file_extension = os.path.splitext(filename)[1]
            content_text = self._extract_text_from_file(file_path, file_extension)
            
            # Generate embedding for the content
            embedding_vector = self._generate_embedding(content_text)

            if self.supabase:
                # Insert into Supabase research_documents table per actual schema
                payload = {
                    "file_name": filename,
                    "file_content": content_text or f"Document: {filename}",
                    "symbol": symbol,
                    "uploaded_at": datetime.now().isoformat()
                }
                try:
                    resp = self.supabase.table("research_documents").insert(payload).execute()
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
                
            query = self.supabase.table("research_documents").select("*")
            # Filter by symbol if provided
            if symbol:
                query = query.eq("symbol", symbol)
            
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
                
            resp = self.supabase.table("research_documents").select("file_content").eq("id", document_id).execute()
            data = resp.data if hasattr(resp, "data") else []
            if data and len(data) > 0:
                return data[0].get("file_content")
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
            resp = self.supabase.table("research_documents").delete().eq("id", document_id).execute()
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
            resp = self.supabase.table("research_documents").select("*").execute()
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
    
    def analyze_document_with_ai(self, document_id: str, symbol: str = None) -> Dict:
        """Analyze document content using AI to extract trading insights"""
        try:
            # Get document content
            content = self.get_document_content(document_id)
            if not content:
                return {"success": False, "error": "Document content not found"}
            
            # Use Groq/Phi for analysis
            from phi.model.groq import Groq
            from phi.agent.agent import Agent
            
            # Initialize Groq model
            groq_model = Groq(
                model="llama-3.1-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY", "")
            )
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze the following research document for trading insights related to {symbol or 'the mentioned stock'}.
            
            Document Content:
            {content[:4000]}  # Limit content to avoid token limits
            
            Please provide:
            1. Key financial metrics mentioned
            2. Bullish signals (positive indicators)
            3. Bearish signals (negative indicators)
            4. Overall sentiment (Bullish/Bearish/Neutral)
            5. Confidence level (1-10)
            6. Key risks mentioned
            7. Investment recommendation (BUY/SELL/HOLD)
            
            Format your response as a structured analysis.
            """
            
            # Get AI analysis
            response = groq_model.generate(analysis_prompt)
            
            return {
                "success": True,
                "analysis": response,
                "document_id": document_id,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_trading_signals(self, document_id: str) -> Dict:
        """Extract specific trading signals from document"""
        try:
            content = self.get_document_content(document_id)
            if not content:
                return {"success": False, "error": "Document content not found"}
            
            # Look for common trading signal keywords
            bullish_keywords = [
                "buy", "bullish", "positive", "growth", "outperform", "upgrade",
                "strong", "beat", "exceed", "increase", "rise", "gain"
            ]
            
            bearish_keywords = [
                "sell", "bearish", "negative", "decline", "underperform", "downgrade",
                "weak", "miss", "fall", "decrease", "drop", "loss"
            ]
            
            content_lower = content.lower()
            
            bullish_signals = [word for word in bullish_keywords if word in content_lower]
            bearish_signals = [word for word in bearish_keywords if word in content_lower]
            
            # Calculate sentiment score
            sentiment_score = len(bullish_signals) - len(bearish_signals)
            
            if sentiment_score > 0:
                overall_sentiment = "Bullish"
            elif sentiment_score < 0:
                overall_sentiment = "Bearish"
            else:
                overall_sentiment = "Neutral"
            
            return {
                "success": True,
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals,
                "sentiment_score": sentiment_score,
                "overall_sentiment": overall_sentiment,
                "confidence": min(abs(sentiment_score) * 2, 10)  # Scale to 1-10
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_document_insights(self, symbol: str = None) -> List[Dict]:
        """Get AI insights for all documents or documents for a specific symbol"""
        try:
            documents = self.get_documents(symbol=symbol)
            insights = []
            
            for doc in documents:
                doc_id = doc.get("id")
                if doc_id:
                    # Get trading signals
                    signals = self.extract_trading_signals(doc_id)
                    
                    # Get AI analysis if available
                    analysis = self.analyze_document_with_ai(doc_id, symbol)
                    
                    insight = {
                        "document_id": doc_id,
                        "filename": doc.get("metadata", {}).get("file_name", "Unknown"),
                        "symbol": doc.get("metadata", {}).get("symbol", symbol),
                        "signals": signals,
                        "analysis": analysis,
                        "uploaded_at": doc.get("created_at", "Unknown")
                    }
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            print(f"Error getting document insights: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        # No database connection to close for Supabase
        pass
