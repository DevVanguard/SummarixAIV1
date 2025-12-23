"""
Mode selector component for extractive/abstractive summarization.
This widget lets users choose between extractive and abstractive modes,
and also select the desired summary length for abstractive mode.
"""

import logging
from enum import Enum

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
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
        Set up all the UI components - mode selection with card-based design.
        Enhanced following HCI principles:
        - Clear visual hierarchy with icons
        - Card-based design for each mode
        - Better affordances with hover states
        """
        layout = QVBoxLayout()
        layout.setSpacing(0)  # No extra spacing, GroupBox handles it
        layout.setContentsMargins(0, 0, 0, 0)  # Consistent margins for alignment
        
        # Main group box for mode selection with modern styling
        group_box = QGroupBox("⚡ Summarization Mode")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(12)  # Better breathing room between cards
        group_layout.setContentsMargins(16, 8, 16, 16)  # Reduced top padding since GroupBox title uses space
        
        # Button group to ensure radio buttons are mutually exclusive
        self.mode_button_group = QButtonGroup(self)
        
        # Extractive mode option with card-like container
        extractive_container = QWidget()
        extractive_container.setObjectName("modeCard")
        extractive_container.setStyleSheet("""
            QWidget#modeCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1f212b, stop:1 #1a1c26);
                border: 2px solid #3a3d4a;
                border-radius: 12px;
            }
            QWidget#modeCard:hover {
                border: 2px solid #7c8ff5;
            }
        """)
        extractive_layout = QVBoxLayout()
        extractive_layout.setSpacing(6)
        extractive_layout.setContentsMargins(12, 10, 12, 10)
        
        # Horizontal layout for radio button and description
        extractive_top_layout = QHBoxLayout()
        extractive_top_layout.setSpacing(12)
        
        self.extractive_radio = QRadioButton("⚡ Extractive")
        self.extractive_radio.setChecked(True)  # Default selection
        self.extractive_radio.toggled.connect(self._on_mode_changed)
        self.extractive_radio.setStyleSheet("""
            QRadioButton {
                font-weight: 600;
                font-size: 10.5pt;
                background: transparent;
                border: none;
                padding: 2px 0px;
            }
        """)
        self.mode_button_group.addButton(self.extractive_radio)
        
        extractive_desc = QLabel("Fast • Key sentences • Original wording")
        extractive_desc.setWordWrap(True)
        extractive_desc.setStyleSheet("""
            QLabel {
                color: #9ca3af;
                font-size: 9pt;
                background: transparent;
                border: none;
            }
        """)
        
        extractive_top_layout.addWidget(self.extractive_radio, 0)
        extractive_top_layout.addWidget(extractive_desc, 1)
        
        # Info label for disabled length selector
        self.length_info_label = QLabel("💡 Length options only available for Abstractive mode")
        self.length_info_label.setWordWrap(True)
        self.length_info_label.setStyleSheet("""
            QLabel {
                color: #7c8ff5;
                font-size: 8pt;
                font-style: italic;
                padding: 4px 0px 0px 0px;
                background: transparent;
                border: none;
            }
        """)
        self.length_info_label.hide()  # Will show when extractive is selected
        
        extractive_layout.addLayout(extractive_top_layout)
        extractive_layout.addWidget(self.length_info_label)
        extractive_container.setLayout(extractive_layout)
        
        # Abstractive mode option with card-like container
        abstractive_container = QWidget()
        abstractive_container.setObjectName("modeCard")
        abstractive_container.setStyleSheet("""
            QWidget#modeCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1f212b, stop:1 #1a1c26);
                border: 2px solid #3a3d4a;
                border-radius: 12px;
            }
            QWidget#modeCard:hover {
                border: 2px solid #7c8ff5;
            }
        """)
        abstractive_layout = QVBoxLayout()
        abstractive_layout.setSpacing(6)
        abstractive_layout.setContentsMargins(12, 10, 12, 10)
        
        # Horizontal layout for radio button, description, and length selector
        abstractive_top_layout = QHBoxLayout()
        abstractive_top_layout.setSpacing(12)
        
        self.abstractive_radio = QRadioButton("🤖 Abstractive")
        self.abstractive_radio.toggled.connect(self._on_mode_changed)
        self.abstractive_radio.setStyleSheet("""
            QRadioButton {
                font-weight: 600;
                font-size: 10.5pt;
                background: transparent;
                border: none;
                padding: 2px 0px;
            }
        """)
        self.mode_button_group.addButton(self.abstractive_radio)
        
        abstractive_desc = QLabel("AI-powered • Natural language • Concise")
        abstractive_desc.setWordWrap(True)
        abstractive_desc.setStyleSheet("""
            QLabel {
                color: #9ca3af;
                font-size: 9pt;
                background: transparent;
                border: none;
            }
        """)
        
        # Summary length selector (only for abstractive mode) - on the right side
        length_label = QLabel("📏 Length:")
        length_label.setStyleSheet("""
            QLabel {
                color: #e8eaed;
                font-size: 9pt;
                font-weight: 500;
                background: transparent;
                border: none;
            }
        """)
        
        self.length_combo = QComboBox()
        self.length_combo.addItems(["Short", "Medium", "Long"])
        self.length_combo.setCurrentText("Medium")  # Default to medium
        self.length_combo.currentTextChanged.connect(self._on_length_changed)
        self.length_combo.setEnabled(False)  # Disabled until abstractive is selected
        self.length_combo.setFixedHeight(34)  # Increased height for visibility
        self.length_combo.setMinimumWidth(100)  # Compact width
        # No need to set styleSheet - it will use the theme
        
        abstractive_top_layout.addWidget(self.abstractive_radio, 0)
        abstractive_top_layout.addWidget(abstractive_desc, 1)
        abstractive_top_layout.addWidget(length_label, 0)
        abstractive_top_layout.addWidget(self.length_combo, 0)
        
        abstractive_layout.addLayout(abstractive_top_layout)
        abstractive_container.setLayout(abstractive_layout)
        
        group_layout.addWidget(extractive_container)
        group_layout.addWidget(abstractive_container)
        
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

