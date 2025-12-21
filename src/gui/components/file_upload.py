"""File upload component with drag-and-drop support."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.utils.validators import validate_file_size, validate_memory_for_file

logger = logging.getLogger(__name__)


class ClickableLabel(QLabel):
    """A clickable QLabel that emits a clicked signal."""
    
    clicked = pyqtSignal()
    
    def mousePressEvent(self, event):
        """Handle mouse press event."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ValidationWorker(QThread):
    """Worker thread for asynchronous file validation."""
    
    check_complete = pyqtSignal(str, str, str)  # name, status, message
    validation_finished = pyqtSignal(bool)  # is_valid
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        """Run validation checks."""
        try:
            from src.core.pdf_processor import PDFProcessor
            from src.utils.config import Config
            from src.utils.validators import validate_file_size, validate_memory_for_file
            
            all_valid = True
            
            # 1. File size validation
            try:
                is_valid, warning, error = validate_file_size(self.file_path)
                if error:
                    logger.error(f"File size validation failed: {error}")
                    self.validation_finished.emit(False)
                    return
                elif warning:
                    logger.warning(f"File size warning: {warning}")
            except Exception as e:
                logger.error(f"Error in file size validation: {str(e)}", exc_info=True)
                self.validation_finished.emit(False)
                return
            
            # 2. PDF format validation
            try:
                with PDFProcessor() as processor:
                    if not processor.load_pdf(self.file_path):
                        logger.error("Failed to load PDF file")
                        self.validation_finished.emit(False)
                        return
                    
                    is_valid, error_msg = processor.validate_pdf()
                    if not is_valid:
                        logger.error(f"PDF validation failed: {error_msg}")
                        self.validation_finished.emit(False)
                        return
                    
                    if processor.is_password_protected():
                        logger.error("PDF is password-protected")
                        self.validation_finished.emit(False)
                        return
                    
                    # 3. Text extractability (quick check - first page only)
                    text_extractable, warning_msg, text_length = processor.check_text_extractable()
                    if not text_extractable:
                        logger.error(f"Text extraction failed: {warning_msg}")
                        self.validation_finished.emit(False)
                        return
                    
                    # 4. Memory check (quick)
                    try:
                        has_memory, mem_warning, mem_error = validate_memory_for_file(self.file_path)
                        if mem_error:
                            logger.error(f"Memory validation failed: {mem_error}")
                            self.validation_finished.emit(False)
                            return
                    except Exception as e:
                        logger.warning(f"Memory check error (continuing anyway): {str(e)}")
                        # Don't fail validation on memory check errors - just warn
                    
                    # All validations passed
                    logger.info(f"File validation passed for: {self.file_path.name}")
                    self.validation_finished.emit(True)
                    return
                    
            except Exception as e:
                logger.error(f"Error in PDF validation: {str(e)}", exc_info=True)
                self.validation_finished.emit(False)
                return
                
        except Exception as e:
            logger.error(f"Unexpected validation error: {str(e)}", exc_info=True)
            self.validation_finished.emit(False)


class FileUploadWidget(QWidget):
    """
    Widget for PDF file upload with drag-and-drop.
    Provides a clean interface for selecting files with the ability to remove selection.
    """
    
    file_selected = pyqtSignal(Path)  # Emitted when a file is selected
    file_cleared = pyqtSignal()  # Emitted when the file selection is cleared
    validation_complete = pyqtSignal(bool)  # Emitted when validation completes (True if valid)
    
    def __init__(self, parent=None):
        """Initialize file upload widget."""
        super().__init__(parent)
        self.selected_file: Optional[Path] = None
        self.validation_worker: Optional[ValidationWorker] = None
        self.validation_status = ""  # "", "validating", "valid", "invalid", "warning"
        self._setup_ui()
        self.setAcceptDrops(True)
    
    def _setup_ui(self):
        """
        Set up the UI components with dynamic visibility.
        Follows HCI principles: efficient space usage, clear visual hierarchy, proper affordances.
        """
        layout = QVBoxLayout()
        layout.setSpacing(8)  # Reduced spacing for efficiency
        layout.setContentsMargins(0, 0, 0, 0)  # Remove internal margins for consistent width
        
        # Instructions label with modern styling (shown when no file selected)
        # Made clickable to browse files
        self.instruction_label = ClickableLabel("📄 Drag and drop a PDF file here\nor click to browse")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.instruction_label.clicked.connect(self._browse_files)
        self.instruction_label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 11.5pt;
                padding: 32px 20px;
                border: 2px dashed #3d3d3d;
                border-radius: 12px;
                background-color: #252525;
                min-height: 80px;
                font-weight: 500;
                /* Enhanced card-like depth */
                border-top: 2px dashed #4d4d4d;
                border-left: 2px dashed #4d4d4d;
                border-bottom: 2px dashed #2d2d2d;
                border-right: 2px dashed #2d2d2d;
            }
            QLabel:hover {
                color: #b0b0b0;
                border: 2px dashed #5d5d5d;
                background-color: #2a2a2a;
                border-top: 2px dashed #60b8ff;
                border-left: 2px dashed #60b8ff;
            }
        """)
        
        # File info container (shown when file is selected)
        self.file_info_container = QWidget()
        file_info_layout = QVBoxLayout()
        file_info_layout.setSpacing(8)
        file_info_layout.setContentsMargins(0, 0, 0, 0)
        
        # File display with icon and info
        file_display_layout = QHBoxLayout()
        file_display_layout.setSpacing(8)  # Reduced spacing for compact layout
        
        # File icon/name label with proper text handling and validation status
        self.file_label = ClickableLabel("No file selected")
        self.file_label.setWordWrap(True)  # Allow wrapping for very long names
        self.file_label.setTextFormat(Qt.TextFormat.RichText)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.file_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.file_label.clicked.connect(self._browse_files)
        self.file_label.setStyleSheet("""
            QLabel {
                color: #4ec9b0;
                font-size: 10pt;
                padding: 10px 12px;
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: 6px;
            }
            QLabel:hover {
                background-color: #323232;
                border: 2px solid #4d4d4d;
            }
        """)
        self.validation_status = ""  # Track validation status: "", "validating", "valid", "invalid", "warning"
        
        # Remove button (X button to clear selection) - compact size with semantic red color
        self.remove_button = QPushButton("✕")
        self.remove_button.setToolTip("Remove selected file")
        self.remove_button.setFixedSize(24, 24)  # Compact size to fit component
        self.remove_button.setMinimumSize(0, 0)  # Override Qt's minimum button size
        self.remove_button.setProperty("buttonType", "danger")  # Apply semantic color
        self.remove_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dc3545, stop:1 #c82333);
                color: white;
                border: 1px solid #bd2130;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e4606d, stop:1 #dc3545);
                border: 1px solid #c82333;
                border-top: 2px solid #ff6b7a;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #c82333, stop:1 #bd2130);
                border: 1px solid #a71e2a;
            }
            QPushButton:focus {
                outline: 2px solid #ff6b7a;
                outline-offset: 1px;
            }
        """)
        self.remove_button.clicked.connect(self.clear_selection)
        self.remove_button.hide()  # Hidden initially
        
        file_display_layout.addWidget(self.file_label, 1)  # Takes remaining space
        file_display_layout.addWidget(self.remove_button, 0)  # Fixed size
        
        file_info_layout.addLayout(file_display_layout)
        self.file_info_container.setLayout(file_info_layout)
        self.file_info_container.hide()  # Hidden initially
        
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.file_info_container)
        
        self.setLayout(layout)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """
        Handle drag enter event - highlight the drop zone when dragging over it.
        Works whether a file is selected or not - allows replacing existing file.
        """
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().endswith('.pdf'):
                event.acceptProposedAction()
                # Highlight the appropriate widget based on current state
                if self.instruction_label.isVisible():
                    # No file selected - highlight instruction label with enhanced hover effect
                    self.instruction_label.setStyleSheet("""
                        QLabel {
                            color: #0078d4;
                            font-size: 12pt;
                            padding: 30px 20px;
                            border: 2px dashed #0078d4;
                            border-radius: 10px;
                            background-color: #1e3a5f;
                            min-height: 80px;
                        }
                    """)
                else:
                    # File already selected - highlight file info container
                    self.file_label.setStyleSheet("""
                        QLabel {
                            color: #0078d4;
                            font-size: 11pt;
                            padding: 12px;
                            background-color: #2d2d2d;
                            border: 2px solid #0078d4;
                            border-radius: 8px;
                        }
                    """)
    
    def dragLeaveEvent(self, event):
        """
        Handle drag leave event - reset the drop zone styling.
        Resets styling for both instruction label and file info container.
        """
        if self.instruction_label.isVisible():
            # Reset instruction label styling
            self.instruction_label.setStyleSheet("""
                QLabel {
                    color: #808080;
                    font-size: 12pt;
                    padding: 30px 20px;
                    border: 2px dashed #3d3d3d;
                    border-radius: 10px;
                    background-color: #2d2d2d;
                    min-height: 80px;
                }
            """)
        else:
            # Reset file label styling
            self.file_label.setStyleSheet("""
                QLabel {
                    color: #4ec9b0;
                    font-size: 11pt;
                    padding: 12px;
                    background-color: #2d2d2d;
                    border: 2px solid #3d3d3d;
                    border-radius: 8px;
                }
            """)
    
    def dropEvent(self, event: QDropEvent):
        """
        Handle drop event - process the dropped file.
        Can replace an existing file if one is already selected.
        Resets the drop zone styling after processing.
        """
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = Path(urls[0].toLocalFile())
                if file_path.suffix.lower() == '.pdf':
                    self._set_file(file_path)
                else:
                    logger.warning(f"Invalid file type: {file_path.suffix}")
            event.acceptProposedAction()
        
        # Reset styling after drop
        if self.instruction_label.isVisible():
            # Reset instruction label styling
            self.instruction_label.setStyleSheet("""
                QLabel {
                    color: #808080;
                    font-size: 12pt;
                    padding: 30px 20px;
                    border: 2px dashed #3d3d3d;
                    border-radius: 10px;
                    background-color: #2d2d2d;
                    min-height: 80px;
                }
            """)
        else:
            # Reset file label styling (file was replaced)
            self.file_label.setStyleSheet("""
                QLabel {
                    color: #4ec9b0;
                    font-size: 11pt;
                    padding: 12px;
                    background-color: #2d2d2d;
                    border: 2px solid #3d3d3d;
                    border-radius: 8px;
                }
            """)
    
    def _browse_files(self):
        """Open file browser dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF File",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            self._set_file(Path(file_path))
    
    def _set_file(self, file_path: Path):
        """
        Set the selected file and update UI to show file info.
        Hides the drag-and-drop placeholder and shows the file info with remove button.
        """
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return
        
        try:
            self.selected_file = file_path
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            
            # Format file size nicely
            if file_size < 1:
                size_str = f"{file_size * 1024:.1f} KB"
            else:
                size_str = f"{file_size:.2f} MB"
            
            # Update file label with nice formatting - use elision for long names
            file_name = file_path.name
            # Use Qt's elision to handle long filenames elegantly
            # Calculate available width dynamically
            font_metrics = self.file_label.fontMetrics()
            # Account for icon, spacing, and remove button (32px + 12px spacing)
            available_width = max(300, self.width() - 100)  # Dynamic width calculation
            elided_name = font_metrics.elidedText(file_name, Qt.TextElideMode.ElideMiddle, available_width)
            
            # File label will be updated by _update_file_label method
            
            # Hide drag-and-drop placeholder
            self.instruction_label.hide()
            
            # Show file info container with remove button
            self.file_info_container.show()
            self.remove_button.show()
            
            logger.info(f"File selected: {file_path}")
            
            # Update file label with validation status
            self._update_file_label(file_path, size_str, "validating")
            
            # Start validation
            self._validate_file(file_path)
            
            # Emit signal (validation will emit validation_complete when done)
            self.file_selected.emit(file_path)
            
        except Exception as e:
            logger.error(f"Error setting file: {str(e)}")
            # Reset UI state on error
            self.clear_selection()
    
    def get_selected_file(self) -> Optional[Path]:
        """Get the currently selected file."""
        return self.selected_file
    
    def clear_selection(self):
        """
        Clear the file selection and reset UI to initial state.
        Shows the drag-and-drop placeholder again and hides file info.
        """
        self.selected_file = None
        
        # Reset file label
        self.file_label.setText("No file selected")
        
        # Show drag-and-drop placeholder again
        self.instruction_label.show()
        
        # Hide file info container
        self.file_info_container.hide()
        self.remove_button.hide()
        
        # Stop validation worker if running
        if self.validation_worker and self.validation_worker.isRunning():
            self.validation_worker.terminate()
            self.validation_worker.wait()
            self.validation_worker = None
        
        self.validation_status = ""
        
        logger.info("File selection cleared")
        
        # Emit signal that file was cleared so other components can react
        self.file_cleared.emit()
    
    def set_enabled(self, enabled: bool):
        """
        Enable or disable all interactive elements in the file upload widget.
        
        Args:
            enabled: True to enable, False to disable
        """
        self.remove_button.setEnabled(enabled)
        self.setAcceptDrops(enabled)  # Enable/disable drag and drop
        # Update cursor based on enabled state
        if enabled:
            self.instruction_label.setCursor(Qt.CursorShape.PointingHandCursor)
            self.file_info_container.setCursor(Qt.CursorShape.PointingHandCursor)
            self.file_label.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.instruction_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.file_info_container.setCursor(Qt.CursorShape.ArrowCursor)
            self.file_label.setCursor(Qt.CursorShape.ArrowCursor)
    
    def _validate_file(self, file_path: Path):
        """
        Validate the selected file asynchronously and update UI with validation status.
        
        Args:
            file_path: Path to the file to validate
        """
        # Stop any existing validation
        if self.validation_worker and self.validation_worker.isRunning():
            self.validation_worker.terminate()
            self.validation_worker.wait()
        
        # Start asynchronous validation
        self.validation_worker = ValidationWorker(file_path)
        self.validation_worker.validation_finished.connect(self._on_validation_finished)
        self.validation_worker.start()
    
    def _on_validation_finished(self, is_valid: bool):
        """Handle validation completion signal."""
        if self.selected_file:
            file_size = self.selected_file.stat().st_size / (1024 * 1024)
            size_str = f"{file_size:.2f} MB" if file_size >= 1 else f"{file_size * 1024:.1f} KB"
            
            # Update validation status
            if is_valid:
                self.validation_status = "valid"
            else:
                self.validation_status = "invalid"
                # Show error message for validation failure
                QMessageBox.warning(
                    self,
                    "Validation Failed",
                    f"File validation failed for: {self.selected_file.name}\n\n"
                    "The file may be:\n"
                    "- Too large (max 100 MB)\n"
                    "- Corrupted or invalid PDF\n"
                    "- Password-protected\n"
                    "- Image-based (scanned document)\n"
                    "- Missing extractable text\n\n"
                    "Please select a different file."
                )
            
            # Update file label with validation status
            self._update_file_label(self.selected_file, size_str, self.validation_status)
        
        self.validation_complete.emit(is_valid)
        if self.validation_worker:
            self.validation_worker.deleteLater()
            self.validation_worker = None
    
    def _update_file_label(self, file_path: Path, size_str: str, status: str = ""):
        """
        Update the file label with file info and validation status icon.
        
        Args:
            file_path: Path to the file
            size_str: Formatted file size string
            status: Validation status ("", "validating", "valid", "invalid", "warning")
        """
        file_name = file_path.name
        font_metrics = self.file_label.fontMetrics()
        available_width = max(300, self.width() - 100)
        elided_name = font_metrics.elidedText(file_name, Qt.TextElideMode.ElideMiddle, available_width)
        
        # Choose status icon
        if status == "validating":
            status_icon = "⏳"
            status_color = "#808080"
        elif status == "valid":
            status_icon = "✓"
            status_color = "#28a745"
        elif status == "invalid":
            status_icon = "✗"
            status_color = "#dc3545"
        elif status == "warning":
            status_icon = "⚠"
            status_color = "#ffc107"
        else:
            status_icon = ""
            status_color = "#4ec9b0"
        
        # Build label text with HTML formatting
        if status_icon:
            label_text = f"📄 {elided_name} <span style='color: {status_color}; font-weight: bold; font-size: 11pt;'>{status_icon}</span><br><span style='color: #808080; font-size: 9pt;'>Size: {size_str}</span>"
        else:
            label_text = f"📄 {elided_name}<br><span style='color: #808080; font-size: 9pt;'>Size: {size_str}</span>"
        
        self.file_label.setText(label_text)
        self.file_label.setToolTip(f"Full path: {file_path}\nSize: {size_str}\nClick to change file")
    
    def is_file_valid(self) -> bool:
        """Check if the currently selected file has passed all validations."""
        return self.validation_status == "valid"

