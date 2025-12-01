"""Mode selector component for extractive/abstractive summarization."""

import logging
from enum import Enum

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class SummarizationMode(Enum):
    """Summarization mode enumeration."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"


class ModeSelectorWidget(QWidget):
    """Widget for selecting summarization mode."""
    
    mode_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize mode selector widget."""
        super().__init__(parent)
        self.current_mode = SummarizationMode.EXTRACTIVE
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Group box
        group_box = QGroupBox("Summarization Mode")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(16)
        
        # Extractive mode
        extractive_layout = QHBoxLayout()
        self.extractive_radio = QRadioButton("Extractive (TextRank)")
        self.extractive_radio.setChecked(True)
        self.extractive_radio.toggled.connect(self._on_mode_changed)
        
        extractive_desc = QLabel(
            "Extracts key sentences from the document.\n"
            "Faster, preserves original wording."
        )
        extractive_desc.setWordWrap(True)
        extractive_desc.setStyleSheet("color: #808080; font-size: 9pt;")
        
        extractive_layout.addWidget(self.extractive_radio)
        extractive_layout.addWidget(extractive_desc)
        extractive_layout.addStretch()
        
        # Abstractive mode
        abstractive_layout = QHBoxLayout()
        self.abstractive_radio = QRadioButton("Abstractive (T5)")
        self.abstractive_radio.toggled.connect(self._on_mode_changed)
        
        abstractive_desc = QLabel(
            "Generates new summary text.\n"
            "Slower, requires model loading, more natural summaries."
        )
        abstractive_desc.setWordWrap(True)
        abstractive_desc.setStyleSheet("color: #808080; font-size: 9pt;")
        
        abstractive_layout.addWidget(self.abstractive_radio)
        abstractive_layout.addWidget(abstractive_desc)
        abstractive_layout.addStretch()
        
        group_layout.addLayout(extractive_layout)
        group_layout.addLayout(abstractive_layout)
        
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        
        self.setLayout(layout)
    
    def _on_mode_changed(self):
        """Handle mode change."""
        if self.extractive_radio.isChecked():
            self.current_mode = SummarizationMode.EXTRACTIVE
            mode_str = "extractive"
        else:
            self.current_mode = SummarizationMode.ABSTRACTIVE
            mode_str = "abstractive"
        
        logger.info(f"Mode changed to: {mode_str}")
        self.mode_changed.emit(mode_str)
    
    def get_mode(self) -> SummarizationMode:
        """Get the currently selected mode."""
        return self.current_mode
    
    def set_mode(self, mode: SummarizationMode):
        """Set the summarization mode."""
        if mode == SummarizationMode.EXTRACTIVE:
            self.extractive_radio.setChecked(True)
        else:
            self.abstractive_radio.setChecked(True)

