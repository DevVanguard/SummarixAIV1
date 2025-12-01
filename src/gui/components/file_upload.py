"""File upload component with drag-and-drop support."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class FileUploadWidget(QWidget):
    """Widget for PDF file upload with drag-and-drop."""
    
    file_selected = pyqtSignal(Path)
    
    def __init__(self, parent=None):
        """Initialize file upload widget."""
        super().__init__(parent)
        self.selected_file: Optional[Path] = None
        self._setup_ui()
        self.setAcceptDrops(True)
    
    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Instructions label
        self.instruction_label = QLabel("Drag and drop a PDF file here")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 11pt;
                padding: 20px;
                border: 2px dashed #3d3d3d;
                border-radius: 8px;
                background-color: #2d2d2d;
            }
        """)
        
        # File info label
        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
                font-size: 9pt;
                padding: 8px;
            }
        """)
        
        # Browse button
        self.browse_button = QPushButton("Browse Files")
        self.browse_button.clicked.connect(self._browse_files)
        
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.file_label)
        layout.addWidget(self.browse_button)
        
        self.setLayout(layout)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().endswith('.pdf'):
                event.acceptProposedAction()
                self.instruction_label.setStyleSheet("""
                    QLabel {
                        color: #0078d4;
                        font-size: 11pt;
                        padding: 20px;
                        border: 2px dashed #0078d4;
                        border-radius: 8px;
                        background-color: #2d2d2d;
                    }
                """)
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.instruction_label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 11pt;
                padding: 20px;
                border: 2px dashed #3d3d3d;
                border-radius: 8px;
                background-color: #2d2d2d;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = Path(urls[0].toLocalFile())
                if file_path.suffix.lower() == '.pdf':
                    self._set_file(file_path)
                else:
                    logger.warning(f"Invalid file type: {file_path.suffix}")
            event.acceptProposedAction()
        
        # Reset style
        self.instruction_label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 11pt;
                padding: 20px;
                border: 2px dashed #3d3d3d;
                border-radius: 8px;
                background-color: #2d2d2d;
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
        """Set the selected file."""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return
        
        self.selected_file = file_path
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        
        self.file_label.setText(
            f"Selected: {file_path.name}\n"
            f"Size: {file_size:.2f} MB"
        )
        self.file_label.setStyleSheet("""
            QLabel {
                color: #4ec9b0;
                font-size: 9pt;
                padding: 8px;
            }
        """)
        
        logger.info(f"File selected: {file_path}")
        self.file_selected.emit(file_path)
    
    def get_selected_file(self) -> Optional[Path]:
        """Get the currently selected file."""
        return self.selected_file
    
    def clear_selection(self):
        """Clear the file selection."""
        self.selected_file = None
        self.file_label.setText("No file selected")
        self.file_label.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
                font-size: 9pt;
                padding: 8px;
            }
        """)

