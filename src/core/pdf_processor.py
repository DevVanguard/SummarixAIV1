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
    
    def validate_pdf(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that the loaded PDF is valid and not corrupted.
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if PDF is valid
            - error_message: Error message if invalid (or None)
        """
        if self.doc is None:
            return False, "No PDF document loaded"
        
        try:
            # Check if document is valid
            if self.doc.is_pdf:
                # Try to access page count (will fail if corrupted)
                page_count = len(self.doc)
                if page_count == 0:
                    return False, "PDF appears to be empty (0 pages)"
                return True, None
            else:
                return False, "File is not a valid PDF format"
        except Exception as e:
            logger.error(f"PDF validation error: {str(e)}")
            return False, f"PDF appears to be corrupted: {str(e)}"
    
    def is_password_protected(self) -> bool:
        """
        Check if the PDF is password-protected.
        
        Returns:
            True if password-protected, False otherwise
        """
        if self.doc is None:
            return False
        
        try:
            # Check if PDF needs password - PyMuPDF sets needs_pass attribute
            # Also try to access a page to verify it's not password-protected
            if hasattr(self.doc, 'needs_pass') and self.doc.needs_pass:
                return True
            
            # Try to access first page - will fail if password-protected
            if len(self.doc) > 0:
                _ = self.doc[0]  # Try to access first page
            return False
        except Exception as e:
            # Check if it's a password-related error
            error_str = str(e).lower()
            if 'password' in error_str or 'encrypted' in error_str or 'permission' in error_str:
                return True
            # For other exceptions, assume not password-protected (might be other issues)
            logger.debug(f"Exception checking password protection (assuming not protected): {str(e)}")
            return False
    
    def check_text_extractable(self) -> Tuple[bool, Optional[str], int]:
        """
        Check if text can be extracted from the PDF.
        
        Returns:
            Tuple of (is_extractable, warning_message, text_length)
            - is_extractable: True if text can be extracted
            - warning_message: Warning if text is minimal (or None)
            - text_length: Number of characters extracted
        """
        if self.doc is None:
            return False, "No PDF document loaded", 0
        
        try:
            # Try to extract text from first few pages
            text_parts = []
            pages_to_check = min(5, len(self.doc))  # Check first 5 pages or all if less
            
            for page_num in range(pages_to_check):
                try:
                    page = self.doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
            
            if not text_parts:
                # No text found - might be image-based
                return False, "No extractable text found. PDF might be image-based (scanned document).", 0
            
            # Get full text length estimate
            sample_text = "\n\n".join(text_parts)
            sample_length = len(sample_text)
            
            # Estimate total text length
            if len(self.doc) > pages_to_check:
                estimated_length = int(sample_length * (len(self.doc) / pages_to_check))
            else:
                estimated_length = sample_length
            
            # Check minimum text length
            from src.utils.config import Config
            if estimated_length < Config.MIN_TEXT_LENGTH:
                warning_msg = (
                    f"Very little text found ({estimated_length} characters). "
                    f"PDF might be mostly images or empty."
                )
                return True, warning_msg, estimated_length
            
            return True, None, estimated_length
            
        except Exception as e:
            logger.error(f"Error checking text extractability: {str(e)}")
            return False, f"Error checking text: {str(e)}", 0
    
    def detect_image_only(self) -> bool:
        """
        Detect if PDF is likely image-only (scanned document).
        
        Returns:
            True if PDF appears to be image-only, False otherwise
        """
        if self.doc is None:
            return False
        
        try:
            # Check first page for text content
            if len(self.doc) == 0:
                return False
            
            first_page = self.doc[0]
            text = first_page.get_text().strip()
            
            # If no text on first page, likely image-based
            if len(text) < 50:
                # Check a few more pages to be sure
                text_found = False
                for page_num in range(min(3, len(self.doc))):
                    page_text = self.doc[page_num].get_text().strip()
                    if len(page_text) > 100:
                        text_found = True
                        break
                
                return not text_found
            
            return False
            
        except Exception:
            return False
    
    def get_validation_info(self) -> Dict[str, any]:
        """
        Get comprehensive validation information about the PDF.
        
        Returns:
            Dictionary with validation results
        """
        info = {
            'is_valid': False,
            'is_password_protected': False,
            'page_count': 0,
            'text_extractable': False,
            'text_length': 0,
            'is_image_only': False,
            'warnings': [],
            'errors': []
        }
        
        if self.doc is None:
            info['errors'].append("No PDF document loaded")
            return info
        
        # Validate PDF structure
        is_valid, error_msg = self.validate_pdf()
        info['is_valid'] = is_valid
        if error_msg:
            info['errors'].append(error_msg)
        
        # Check password protection
        info['is_password_protected'] = self.is_password_protected()
        if info['is_password_protected']:
            info['errors'].append("PDF is password-protected")
        
        # Get page count
        info['page_count'] = self.get_page_count()
        
        # Check text extractability
        text_extractable, warning_msg, text_length = self.check_text_extractable()
        info['text_extractable'] = text_extractable
        info['text_length'] = text_length
        if warning_msg:
            info['warnings'].append(warning_msg)
        if not text_extractable:
            info['errors'].append("Text cannot be extracted from PDF")
        
        # Detect image-only
        info['is_image_only'] = self.detect_image_only()
        if info['is_image_only']:
            info['warnings'].append("PDF appears to be image-based (scanned document)")
        
        # Check page count warning
        from src.utils.config import Config
        if info['page_count'] > Config.MAX_PAGES_WARNING:
            info['warnings'].append(
                f"Large document ({info['page_count']} pages). Processing may take longer."
            )
        
        return info
    
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

