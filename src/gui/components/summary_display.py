"""Summary display component with export functionality."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

try:
    from docx import Document
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    logger.warning("python-docx or reportlab not available. Export features may be limited.")


class SummaryDisplayWidget(QWidget):
    """Widget for displaying and exporting summaries."""
    
    def __init__(self, parent=None):
        """Initialize summary display widget."""
        super().__init__(parent)
        self.summary_text = ""
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Header with stats and actions
        header_layout = QHBoxLayout()
        
        # Stats label
        self.stats_label = QLabel("Word count: 0 | Characters: 0")
        self.stats_label.setStyleSheet("color: #808080; font-size: 9pt;")
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        self.copy_button = QPushButton("Copy")
        self.copy_button.clicked.connect(self._copy_to_clipboard)
        self.copy_button.setEnabled(False)
        
        self.export_txt_button = QPushButton("Export TXT")
        self.export_txt_button.clicked.connect(self._export_txt)
        self.export_txt_button.setEnabled(False)
        
        self.export_pdf_button = QPushButton("Export PDF")
        self.export_pdf_button.clicked.connect(self._export_pdf)
        self.export_pdf_button.setEnabled(False)
        
        self.export_docx_button = QPushButton("Export DOCX")
        self.export_docx_button.clicked.connect(self._export_docx)
        self.export_docx_button.setEnabled(False)
        
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.export_txt_button)
        button_layout.addWidget(self.export_pdf_button)
        button_layout.addWidget(self.export_docx_button)
        
        header_layout.addWidget(self.stats_label)
        header_layout.addStretch()
        header_layout.addLayout(button_layout)
        
        # Text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                line-height: 1.5;
                padding: 12px;
            }
        """)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.text_display)
        
        self.setLayout(layout)
    
    def set_summary(self, text: str):
        """
        Set the summary text to display.
        
        Args:
            text: Summary text
        """
        self.summary_text = text
        self.text_display.setPlainText(text)
        self._update_stats()
        
        # Enable buttons
        has_text = bool(text.strip())
        self.copy_button.setEnabled(has_text)
        self.export_txt_button.setEnabled(has_text)
        self.export_pdf_button.setEnabled(has_text)
        self.export_docx_button.setEnabled(has_text)
    
    def get_summary(self) -> str:
        """Get the current summary text."""
        return self.summary_text
    
    def clear_summary(self):
        """Clear the summary display."""
        self.summary_text = ""
        self.text_display.clear()
        self._update_stats()
        
        # Disable buttons
        self.copy_button.setEnabled(False)
        self.export_txt_button.setEnabled(False)
        self.export_pdf_button.setEnabled(False)
        self.export_docx_button.setEnabled(False)
    
    def _update_stats(self):
        """Update word and character count statistics."""
        text = self.summary_text
        word_count = len(text.split()) if text else 0
        char_count = len(text) if text else 0
        
        self.stats_label.setText(
            f"Word count: {word_count:,} | Characters: {char_count:,}"
        )
    
    def _copy_to_clipboard(self):
        """Copy summary to clipboard."""
        if self.summary_text:
            clipboard = self.text_display.clipboard()
            clipboard.setText(self.summary_text)
            logger.info("Summary copied to clipboard")
    
    def _export_txt(self):
        """Export summary to TXT file."""
        if not self.summary_text:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Summary as TXT",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.summary_text)
                logger.info(f"Summary exported to TXT: {file_path}")
            except Exception as e:
                logger.error(f"Error exporting to TXT: {str(e)}")
    
    def _export_pdf(self):
        """Export summary to PDF file."""
        if not self.summary_text:
            return
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except ImportError:
            logger.error("reportlab not available for PDF export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Summary as PDF",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            try:
                c = canvas.Canvas(file_path, pagesize=letter)
                width, height = letter
                
                # Set up text
                c.setFont("Helvetica", 12)
                y = height - 50
                line_height = 14
                margin = 50
                max_width = width - 2 * margin
                
                # Split text into lines
                words = self.summary_text.split()
                current_line = ""
                
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    text_width = c.stringWidth(test_line, "Helvetica", 12)
                    
                    if text_width > max_width and current_line:
                        c.drawString(margin, y, current_line)
                        y -= line_height
                        if y < margin:
                            c.showPage()
                            y = height - margin
                        current_line = word
                    else:
                        current_line = test_line
                
                if current_line:
                    c.drawString(margin, y, current_line)
                
                c.save()
                logger.info(f"Summary exported to PDF: {file_path}")
            except Exception as e:
                logger.error(f"Error exporting to PDF: {str(e)}")
    
    def _export_docx(self):
        """Export summary to DOCX file."""
        if not self.summary_text:
            return
        
        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx not available for DOCX export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Summary as DOCX",
            "",
            "Word Documents (*.docx);;All Files (*)"
        )
        
        if file_path:
            try:
                doc = Document()
                doc.add_heading('Summary', 0)
                doc.add_paragraph(self.summary_text)
                doc.save(file_path)
                logger.info(f"Summary exported to DOCX: {file_path}")
            except Exception as e:
                logger.error(f"Error exporting to DOCX: {str(e)}")

