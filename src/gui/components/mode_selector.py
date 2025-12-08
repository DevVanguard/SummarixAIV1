"""
Mode selector component for extractive/abstractive summarization.
This widget lets users choose between extractive and abstractive modes,
and also select the desired summary length for abstractive mode.
"""

import logging
from enum import Enum

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
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
    """
    Widget for selecting summarization mode and options.
    Users can choose between extractive and abstractive modes,
    and for abstractive mode, they can select summary length.
    """
    
    mode_changed = pyqtSignal(str)
    length_preset_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize mode selector widget with default settings."""
        super().__init__(parent)
        self.current_mode = SummarizationMode.EXTRACTIVE
        self.current_length_preset = "medium"  # Default to medium length
        self._setup_ui()
    
    def _setup_ui(self):
        """
        Set up all the UI components - mode selection and length options.
        Efficient layout following HCI principles.
        """
        layout = QVBoxLayout()
        layout.setSpacing(8)  # Tighter spacing
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main group box for mode selection
        group_box = QGroupBox("Summarization Mode")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)  # Reduced spacing
        group_layout.setContentsMargins(12, 12, 12, 12)  # Tighter margins
        
        # Extractive mode option
        extractive_layout = QHBoxLayout()
        self.extractive_radio = QRadioButton("Extractive (TextRank)")
        self.extractive_radio.setChecked(True)  # Default selection
        self.extractive_radio.toggled.connect(self._on_mode_changed)
        
        extractive_desc = QLabel(
            "Extracts key sentences from the document.\n"
            "Faster, preserves original wording."
        )
        extractive_desc.setWordWrap(True)
        extractive_desc.setStyleSheet("color: #808080; font-size: 9pt;")
        
        # Info label for disabled length selector
        self.length_info_label = QLabel("(Length options only available for Abstractive mode)")
        self.length_info_label.setWordWrap(True)
        self.length_info_label.setStyleSheet("color: #6c757d; font-size: 8pt; font-style: italic; padding: 2px 0px;")
        self.length_info_label.hide()  # Will show when extractive is selected
        
        extractive_layout.addWidget(self.extractive_radio)
        extractive_layout.addWidget(extractive_desc)
        extractive_layout.addStretch()
        
        # Abstractive mode option
        abstractive_layout = QVBoxLayout()
        abstractive_header_layout = QHBoxLayout()
        
        self.abstractive_radio = QRadioButton("Abstractive (T5)")
        self.abstractive_radio.toggled.connect(self._on_mode_changed)
        
        abstractive_desc = QLabel(
            "Generates new summary text using AI.\n"
            "Slower, requires model loading, produces more natural summaries."
        )
        abstractive_desc.setWordWrap(True)
        abstractive_desc.setStyleSheet("color: #808080; font-size: 9pt;")
        
        abstractive_header_layout.addWidget(self.abstractive_radio)
        abstractive_header_layout.addWidget(abstractive_desc)
        abstractive_header_layout.addStretch()
        
        # Summary length selector (only visible for abstractive mode) - compact
        length_layout = QHBoxLayout()
        length_layout.setSpacing(8)
        length_label = QLabel("Length:")
        length_label.setStyleSheet("color: #e0e0e0; font-size: 9pt; padding: 2px 0px;")
        
        self.length_combo = QComboBox()
        self.length_combo.addItems(["Short", "Medium", "Long"])
        self.length_combo.setCurrentText("Medium")  # Default to medium
        self.length_combo.currentTextChanged.connect(self._on_length_changed)
        self.length_combo.setEnabled(False)  # Disabled until abstractive is selected
        self.length_combo.setFixedHeight(28)  # Compact height
        self.length_combo.setStyleSheet("""
            QComboBox {
                font-size: 9pt;
                padding: 4px 8px;
            }
            QComboBox:disabled {
                background-color: #2d2d2d;
                color: #6c757d;
                border-color: #2d2d2d;
                opacity: 0.6;
            }
        """)
        
        length_layout.addWidget(length_label)
        length_layout.addWidget(self.length_combo)
        length_layout.addStretch()
        
        abstractive_layout.addLayout(abstractive_header_layout)
        abstractive_layout.addLayout(length_layout)
        abstractive_layout.addWidget(self.length_info_label)
        
        group_layout.addLayout(extractive_layout)
        group_layout.addLayout(abstractive_layout)
        
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        
        self.setLayout(layout)
    
    def _on_mode_changed(self):
        """
        Handle when user changes the summarization mode.
        Enables/disables the length selector based on mode.
        """
        if self.extractive_radio.isChecked():
            self.current_mode = SummarizationMode.EXTRACTIVE
            mode_str = "extractive"
            # Length selector doesn't apply to extractive mode - gray it out
            self.length_combo.setEnabled(False)
            self.length_info_label.show()  # Show info message
        else:
            self.current_mode = SummarizationMode.ABSTRACTIVE
            mode_str = "abstractive"
            # Enable length selector for abstractive mode
            self.length_combo.setEnabled(True)
            self.length_info_label.hide()  # Hide info message
        
        logger.info(f"Mode changed to: {mode_str}")
        self.mode_changed.emit(mode_str)
    
    def _on_length_changed(self, length_text: str):
        """
        Handle when user changes the summary length preset.
        
        Args:
            length_text: The selected length ("Short", "Medium", or "Long")
        """
        # Convert to lowercase for consistency
        preset = length_text.lower()
        self.current_length_preset = preset
        logger.info(f"Summary length preset changed to: {preset}")
        self.length_preset_changed.emit(preset)
    
    def get_mode(self) -> SummarizationMode:
        """Get the currently selected summarization mode."""
        return self.current_mode
    
    def get_length_preset(self) -> str:
        """
        Get the currently selected summary length preset.
        
        Returns:
            "short", "medium", or "long"
        """
        return self.current_length_preset
    
    def set_mode(self, mode: SummarizationMode):
        """
        Programmatically set the summarization mode.
        
        Args:
            mode: The mode to set (EXTRACTIVE or ABSTRACTIVE)
        """
        if mode == SummarizationMode.EXTRACTIVE:
            self.extractive_radio.setChecked(True)
        else:
            self.abstractive_radio.setChecked(True)
    
    def set_enabled(self, enabled: bool):
        """
        Enable or disable all interactive elements in the mode selector.
        
        Args:
            enabled: True to enable, False to disable
        """
        self.extractive_radio.setEnabled(enabled)
        self.abstractive_radio.setEnabled(enabled)
        # Only enable length combo if abstractive is selected and enabled
        if enabled:
            self.length_combo.setEnabled(enabled and self.abstractive_radio.isChecked())
        else:
            self.length_combo.setEnabled(False)

