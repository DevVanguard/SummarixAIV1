"""PDF text extraction module using PyMuPDF."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF (fitz) is required. Install it with: pip install PyMuPDF"
    )

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handle PDF text extraction and processing."""
    
    def __init__(self):
        """Initialize PDF processor."""
        self.doc: Optional[fitz.Document] = None
    
    def load_pdf(self, file_path: Path) -> bool:
        """
        Load a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not file_path.exists():
                logger.error(f"PDF file not found: {file_path}")
                return False
            
            if not file_path.suffix.lower() == '.pdf':
                logger.error(f"File is not a PDF: {file_path}")
                return False
            
            self.doc = fitz.open(str(file_path))
            logger.info(f"Successfully loaded PDF: {file_path.name} ({len(self.doc)} pages)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return False
    
    def extract_text(self) -> Optional[str]:
        """
        Extract all text from the loaded PDF.
        
        Returns:
            Extracted text as string, or None if extraction failed
        """
        if self.doc is None:
            logger.error("No PDF document loaded")
            return None
        
        try:
            text_parts = []
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def extract_text_with_pages(self) -> Optional[Dict[int, str]]:
        """
        Extract text with page boundaries preserved.
        
        Returns:
            Dictionary mapping page numbers to text, or None if extraction failed
        """
        if self.doc is None:
            logger.error("No PDF document loaded")
            return None
        
        try:
            page_texts = {}
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    page_texts[page_num + 1] = page_text
            
            logger.info(f"Extracted text from {len(page_texts)} pages")
            return page_texts
            
        except Exception as e:
            logger.error(f"Error extracting text with pages: {str(e)}")
            return None
    
    def get_metadata(self) -> Optional[Dict[str, str]]:
        """
        Extract PDF metadata.
        
        Returns:
            Dictionary with metadata, or None if extraction failed
        """
        if self.doc is None:
            logger.error("No PDF document loaded")
            return None
        
        try:
            metadata = self.doc.metadata
            return {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'pages': str(len(self.doc)),
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return None
    
    def get_page_count(self) -> int:
        """
        Get the number of pages in the PDF.
        
        Returns:
            Number of pages, or 0 if no document loaded
        """
        if self.doc is None:
            return 0
        return len(self.doc)
    
    def close(self) -> None:
        """Close the PDF document."""
        if self.doc is not None:
            self.doc.close()
            self.doc = None
            logger.debug("PDF document closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

