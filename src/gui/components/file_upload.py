"""File upload component with drag-and-drop support."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class FileUploadWidget(QWidget):
    """
    Widget for PDF file upload with drag-and-drop.
    Provides a clean interface for selecting files with the ability to remove selection.
    """
    
    file_selected = pyqtSignal(Path)  # Emitted when a file is selected
    file_cleared = pyqtSignal()  # Emitted when the file selection is cleared
    
    def __init__(self, parent=None):
        """Initialize file upload widget."""
        super().__init__(parent)
        self.selected_file: Optional[Path] = None
        self._setup_ui()
        self.setAcceptDrops(True)
    
    def _setup_ui(self):
        """
        Set up the UI components with dynamic visibility.
        Follows HCI principles: efficient space usage, clear visual hierarchy, proper affordances.
        """
        layout = QVBoxLayout()
        layout.setSpacing(8)  # Reduced spacing for efficiency
        layout.setContentsMargins(16, 12, 16, 12)  # Tighter margins
        
        # Instructions label with modern styling (shown when no file selected)
        # Enhanced with better dashed border and hover effect
        self.instruction_label = QLabel("📄 Drag and drop a PDF file here\nor click Browse Files")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 11pt;
                padding: 24px 16px;
                border: 2px dashed #3d3d3d;
                border-radius: 8px;
                background-color: #2d2d2d;
                min-height: 60px;
            }
            QLabel:hover {
                color: #a0a0a0;
                border: 2px dashed #5d5d5d;
                background-color: #323232;
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
        
        # File icon/name label with proper text handling
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)  # Allow wrapping for very long names
        self.file_label.setTextFormat(Qt.TextFormat.PlainText)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.file_label.setStyleSheet("""
            QLabel {
                color: #4ec9b0;
                font-size: 10pt;
                padding: 10px 12px;
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: 6px;
            }
        """)
        
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
        
        # Browse button (always visible, but text changes) - compact professional size
        self.browse_button = QPushButton("Browse Files")
        self.browse_button.clicked.connect(self._browse_files)
        self.browse_button.setFixedHeight(36)  # Compact height following HCI principles
        self.browse_button.setStyleSheet("""
            QPushButton {
                font-size: 10pt;
                font-weight: 500;
                padding: 8px 16px;
                border-radius: 6px;
            }
        """)
        
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.file_info_container)
        layout.addWidget(self.browse_button)
        
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
            
            self.file_label.setText(
                f"📄 {elided_name}\n"
                f"   Size: {size_str}"
            )
            self.file_label.setToolTip(f"Full path: {file_path}\nSize: {size_str}")  # Show full info on hover
            
            # Hide drag-and-drop placeholder
            self.instruction_label.hide()
            
            # Show file info container with remove button
            self.file_info_container.show()
            self.remove_button.show()
            
            # Update browse button text to indicate they can change file
            self.browse_button.setText("Change File")
            
            logger.info(f"File selected: {file_path}")
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
        
        # Reset browse button text
        self.browse_button.setText("Browse Files")
        
        logger.info("File selection cleared")
        
        # Emit signal that file was cleared so other components can react
        self.file_cleared.emit()
    
    def set_enabled(self, enabled: bool):
        """
        Enable or disable all interactive elements in the file upload widget.
        
        Args:
            enabled: True to enable, False to disable
        """
        self.browse_button.setEnabled(enabled)
        self.remove_button.setEnabled(enabled)
        self.setAcceptDrops(enabled)  # Enable/disable drag and drop

