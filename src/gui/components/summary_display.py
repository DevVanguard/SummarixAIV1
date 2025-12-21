"""
Summary display component with export functionality.
This widget shows the generated summary and provides options to copy or export it
in various formats (TXT, PDF, DOCX).
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QClipboard
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
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
        """
        Set up the UI components following HCI principles.
        Efficient layout, proper text handling, compact controls.
        """
        layout = QVBoxLayout()
        layout.setSpacing(8)  # Tighter spacing for efficiency
        layout.setContentsMargins(0, 0, 0, 0)  # Consistent margins for alignment
        
        # Header with stats and actions - compact and efficient
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        header_layout.setContentsMargins(0, 0, 0, 4)
        
        # Stats label - compact display (hidden initially when no file is selected)
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #808080; font-size: 9pt; padding: 4px 0px;")
        self.stats_label.setMinimumWidth(120)  # Fixed width to prevent layout shifts
        self.stats_label.hide()  # Hide initially until a file is selected
        
        # Action buttons - compact and grouped
        button_layout = QHBoxLayout()
        button_layout.setSpacing(6)  # Tighter button spacing
        
        # Compact button styling
        button_style = """
            QPushButton {
                font-size: 9pt;
                font-weight: 500;
                padding: 6px 12px;
                min-height: 28px;
                border-radius: 5px;
            }
        """
        
        self.copy_button = QPushButton("📋 Copy")
        self.copy_button.clicked.connect(self._copy_to_clipboard)
        self.copy_button.setEnabled(False)
        self.copy_button.setStyleSheet(button_style)
        
        self.export_txt_button = QPushButton("💾 TXT")
        self.export_txt_button.setToolTip("Export as Text File")
        self.export_txt_button.clicked.connect(self._export_txt)
        self.export_txt_button.setEnabled(False)
        self.export_txt_button.setStyleSheet(button_style)
        
        self.export_pdf_button = QPushButton("📄 PDF")
        self.export_pdf_button.setToolTip("Export as PDF")
        self.export_pdf_button.clicked.connect(self._export_pdf)
        self.export_pdf_button.setEnabled(False)
        self.export_pdf_button.setStyleSheet(button_style)
        
        self.export_docx_button = QPushButton("📝 DOCX")
        self.export_docx_button.setToolTip("Export as Word Document")
        self.export_docx_button.clicked.connect(self._export_docx)
        self.export_docx_button.setEnabled(False)
        self.export_docx_button.setStyleSheet(button_style)
        
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.export_txt_button)
        button_layout.addWidget(self.export_pdf_button)
        button_layout.addWidget(self.export_docx_button)
        
        header_layout.addWidget(self.stats_label)
        header_layout.addStretch()
        header_layout.addLayout(button_layout)
        
        # Text display with enhanced professional styling - card-like appearance
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)  # Wrap at widget width
        self.text_display.setStyleSheet("""
            QTextEdit {
                font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.7;
                padding: 16px 18px;
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 10px;
                color: #e0e0e0;
                /* Enhanced depth effect */
                border-top: 1px solid #4d4d4d;
                border-left: 1px solid #4d4d4d;
                border-bottom: 2px solid #1a1a1a;
                border-right: 2px solid #1a1a1a;
            }
        """)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.text_display, 1)  # Give text display stretch factor
        
        self.setLayout(layout)
        
        # Make text display expand to use available space
        self.text_display.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
    
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
        """
        Update word and character count statistics.
        Compact format for efficient space usage.
        Hide stats when no file is selected (counts are 0).
        """
        text = self.summary_text
        word_count = len(text.split()) if text else 0
        char_count = len(text) if text else 0
        
        # Hide stats label when no file is selected (both counts are 0)
        if word_count == 0 and char_count == 0:
            self.stats_label.setText("")
            self.stats_label.hide()
        else:
            # Compact format following HCI principles
            self.stats_label.setText(
                f"Words: {word_count:,} | Chars: {char_count:,}"
            )
            self.stats_label.show()
    
    def _copy_to_clipboard(self):
        """
        Copy summary text to system clipboard.
        Shows a brief message if successful.
        """
        if not self.summary_text:
            logger.warning("Attempted to copy empty summary")
            return
        
        try:
            # Get the application's clipboard instance
            clipboard = QApplication.clipboard()
            clipboard.setText(self.summary_text)
            logger.info("Summary copied to clipboard successfully")
            
            # Show brief feedback (optional - could use a status bar message instead)
            # For now, we'll just log it
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {str(e)}")
            # Show error message to user
            QMessageBox.warning(
                self,
                "Copy Failed",
                f"Failed to copy summary to clipboard:\n{str(e)}"
            )
    
    def _export_txt(self):
        """
        Export summary to a plain text file.
        Opens a file dialog for the user to choose save location.
        """
        if not self.summary_text:
            logger.warning("Attempted to export empty summary")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Summary as TXT",
                "summary.txt",  # Default filename
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                # Ensure .txt extension if not provided
                if not file_path.endswith('.txt'):
                    file_path += '.txt'
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.summary_text)
                    logger.info(f"Summary exported to TXT: {file_path}")
                except PermissionError:
                    error_msg = f"Permission denied. Cannot write to:\n{file_path}\n\nFile may be open in another program."
                    logger.error(error_msg)
                    QMessageBox.warning(self, "Export Failed", error_msg)
                except OSError as e:
                    error_msg = f"Failed to write file:\n{str(e)}"
                    logger.error(error_msg)
                    QMessageBox.warning(self, "Export Failed", error_msg)
                except Exception as e:
                    error_msg = f"Unexpected error exporting to TXT:\n{str(e)}"
                    logger.error(error_msg, exc_info=True)
                    QMessageBox.warning(self, "Export Failed", error_msg)
        except Exception as e:
            logger.error(f"Error in export dialog: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to open export dialog:\n{str(e)}")
    
    def _export_pdf(self):
        """
        Export summary to a PDF file.
        Uses reportlab library to create a properly formatted PDF document.
        """
        if not self.summary_text:
            logger.warning("Attempted to export empty summary to PDF")
            return
        
        # Check if reportlab is available
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
        except ImportError:
            error_msg = (
                "PDF export requires the 'reportlab' library.\n\n"
                "Please install it with: pip install reportlab"
            )
            logger.error("reportlab not available for PDF export")
            QMessageBox.warning(self, "Export Unavailable", error_msg)
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Summary as PDF",
                "summary.pdf",  # Default filename
                "PDF Files (*.pdf);;All Files (*)"
            )
            
            if file_path:
                # Ensure .pdf extension
                if not file_path.endswith('.pdf'):
                    file_path += '.pdf'
                
                try:
                    # Create PDF canvas with standard letter size
                    c = canvas.Canvas(file_path, pagesize=letter)
                    width, height = letter
                    
                    # Set up margins and formatting
                    margin = 50
                    top_margin = height - 50
                    bottom_margin = margin
                    max_width = width - 2 * margin
                    
                    # Title
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(margin, top_margin, "Summary")
                    
                    # Summary text with proper word wrapping
                    c.setFont("Helvetica", 11)
                    y = top_margin - 30
                    line_height = 14
                    
                    # Split text into words and wrap lines
                    words = self.summary_text.split()
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        text_width = c.stringWidth(test_line, "Helvetica", 11)
                        
                        # If line is too long, draw current line and start new one
                        if text_width > max_width and current_line:
                            c.drawString(margin, y, current_line)
                            y -= line_height
                            
                            # Check if we need a new page
                            if y < bottom_margin:
                                c.showPage()
                                y = height - margin
                            
                            current_line = word
                        else:
                            current_line = test_line
                    
                    # Draw the last line
                    if current_line:
                        c.drawString(margin, y, current_line)
                    
                    # Save the PDF
                    c.save()
                    logger.info(f"Summary exported to PDF: {file_path}")
                    
                except PermissionError:
                    error_msg = f"Permission denied. Cannot write to:\n{file_path}\n\nFile may be open in another program."
                    logger.error(error_msg)
                    QMessageBox.warning(self, "Export Failed", error_msg)
                except Exception as e:
                    error_msg = f"Error creating PDF:\n{str(e)}"
                    logger.error(error_msg, exc_info=True)
                    QMessageBox.warning(self, "Export Failed", error_msg)
        except Exception as e:
            logger.error(f"Error in PDF export dialog: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to export PDF:\n{str(e)}")
    
    def _export_docx(self):
        """
        Export summary to a Microsoft Word document (DOCX format).
        Uses python-docx library to create a properly formatted document.
        """
        if not self.summary_text:
            logger.warning("Attempted to export empty summary to DOCX")
            return
        
        # Check if python-docx is available
        try:
            from docx import Document
        except ImportError:
            error_msg = (
                "DOCX export requires the 'python-docx' library.\n\n"
                "Please install it with: pip install python-docx"
            )
            logger.error("python-docx not available for DOCX export")
            QMessageBox.warning(self, "Export Unavailable", error_msg)
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Summary as DOCX",
                "summary.docx",  # Default filename
                "Word Documents (*.docx);;All Files (*)"
            )
            
            if file_path:
                # Ensure .docx extension
                if not file_path.endswith('.docx'):
                    file_path += '.docx'
                
                try:
                    # Create a new Word document
                    doc = Document()
                    
                    # Add title
                    doc.add_heading('Summary', 0)
                    
                    # Add the summary text as a paragraph
                    # The library handles formatting automatically
                    doc.add_paragraph(self.summary_text)
                    
                    # Save the document
                    doc.save(file_path)
                    logger.info(f"Summary exported to DOCX: {file_path}")
                    
                except PermissionError:
                    error_msg = f"Permission denied. Cannot write to:\n{file_path}\n\nFile may be open in another program."
                    logger.error(error_msg)
                    QMessageBox.warning(self, "Export Failed", error_msg)
                except Exception as e:
                    error_msg = f"Error creating DOCX file:\n{str(e)}"
                    logger.error(error_msg, exc_info=True)
                    QMessageBox.warning(self, "Export Failed", error_msg)
        except Exception as e:
            logger.error(f"Error in DOCX export dialog: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to export DOCX:\n{str(e)}")

