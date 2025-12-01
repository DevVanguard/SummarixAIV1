"""Progress indicator component for showing processing status."""

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class ProgressIndicatorWidget(QWidget):
    """Widget for displaying progress and status."""
    
    def __init__(self, parent=None):
        """Initialize progress indicator widget."""
        super().__init__(parent)
        self._setup_ui()
        self.hide()
    
    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #e0e0e0; font-size: 9pt;")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
    
    def show_progress(self, message: str = "Processing..."):
        """Show progress indicator."""
        self.status_label.setText(message)
        self.progress_bar.setValue(0)
        self.show()
    
    def update_progress(self, value: int, message: Optional[str] = None):
        """
        Update progress value.
        
        Args:
            value: Progress value (0-100)
            message: Optional status message
        """
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
    
    def set_indeterminate(self, message: str = "Processing..."):
        """Set progress bar to indeterminate mode."""
        self.status_label.setText(message)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
    
    def set_determinate(self, max_value: int = 100):
        """Set progress bar to determinate mode."""
        self.progress_bar.setRange(0, max_value)
        self.progress_bar.setValue(0)
    
    def hide_progress(self):
        """Hide progress indicator."""
        self.hide()
        self.progress_bar.setValue(0)
        self.status_label.setText("")

