"""Validation status widget for displaying file validation checks."""

import logging
from typing import Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class ValidationStatusWidget(QWidget):
    """
    Widget for displaying validation checks with visual status indicators.
    Shows checkmarks, warnings, and errors for each validation step.
    """
    
    # Status constants
    STATUS_PENDING = "pending"
    STATUS_PASSED = "passed"
    STATUS_WARNING = "warning"
    STATUS_FAILED = "failed"
    
    def __init__(self, parent=None):
        """Initialize validation status widget."""
        super().__init__(parent)
        self.checks: Dict[str, Dict] = {}  # Store check data
        self.check_labels: Dict[str, QLabel] = {}  # Store label widgets
        self._setup_ui()
        self.hide()  # Hidden by default
    
    def _setup_ui(self):
        """Set up the UI components with sleek, professional styling."""
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(12, 10, 12, 10)
        
        # Main container with modern card-like appearance
        self.setStyleSheet("""
            QWidget {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
            }
        """)
        
        # Header label with icon
        self.header_label = QLabel("🔍 Validating...")
        self.header_label.setStyleSheet("""
            QLabel {
                color: #4ec9b0;
                font-size: 10pt;
                font-weight: 600;
                padding: 4px 0px;
                background-color: transparent;
            }
        """)
        layout.addWidget(self.header_label)
        
        # Container for validation checks
        self.checks_container = QWidget()
        self.checks_container.setStyleSheet("background-color: transparent;")
        self.checks_layout = QVBoxLayout()
        self.checks_layout.setSpacing(6)
        self.checks_layout.setContentsMargins(8, 4, 0, 0)
        self.checks_container.setLayout(self.checks_layout)
        
        layout.addWidget(self.checks_container)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def add_check(self, name: str, status: str = STATUS_PENDING, message: str = ""):
        """
        Add or update a validation check.
        
        Args:
            name: Name of the validation check
            status: Status (pending, passed, warning, failed)
            message: Status message to display
        """
        if name in self.check_labels:
            # Update existing check
            self.update_check(name, status, message)
        else:
            # Create new check label
            check_label = QLabel()
            check_label.setWordWrap(True)
            check_label.setTextFormat(Qt.TextFormat.PlainText)
            check_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            
            self.checks[name] = {
                'status': status,
                'message': message
            }
            self.check_labels[name] = check_label
            self.checks_layout.addWidget(check_label)
            
            # Update display
            self._update_check_display(name, status, message)
    
    def update_check(self, name: str, status: str, message: str = ""):
        """
        Update an existing validation check.
        
        Args:
            name: Name of the validation check
            status: New status (pending, passed, warning, failed)
            message: Status message to display
        """
        if name not in self.check_labels:
            # Create if doesn't exist
            self.add_check(name, status, message)
            return
        
        # Update stored data
        if name in self.checks:
            self.checks[name]['status'] = status
            self.checks[name]['message'] = message
        
        # Update display
        self._update_check_display(name, status, message)
    
    def _update_check_display(self, name: str, status: str, message: str):
        """Update the visual display of a check with sleek styling."""
        if name not in self.check_labels:
            return
        
        check_label = self.check_labels[name]
        
        # Choose icon and color based on status
        if status == self.STATUS_PENDING:
            icon = "⏳"
            color = "#808080"
            bg_color = "#2d2d2d"
            status_text = "Checking..."
        elif status == self.STATUS_PASSED:
            icon = "✓"
            color = "#28a745"
            bg_color = "#1e3a2e"
            status_text = "OK"
        elif status == self.STATUS_WARNING:
            icon = "⚠"
            color = "#ffc107"
            bg_color = "#3a3a1e"
            status_text = "Warning"
        elif status == self.STATUS_FAILED:
            icon = "✗"
            color = "#dc3545"
            bg_color = "#3a1e1e"
            status_text = "Error"
        else:
            icon = "•"
            color = "#808080"
            bg_color = "#2d2d2d"
            status_text = ""
        
        # Build display text - compact format
        if message and len(message) > 40:
            # Truncate long messages
            message = message[:37] + "..."
        
        display_text = f"{icon} <b>{name}</b>"
        if status_text and status != self.STATUS_PASSED:
            display_text += f" <span style='color: {color};'>{status_text}</span>"
        if message:
            display_text += f"<br><span style='color: #a0a0a0; font-size: 8.5pt;'>{message}</span>"
        
        # Apply sleek styling with background
        check_label.setText(display_text)
        check_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 9pt;
                padding: 6px 10px;
                background-color: {bg_color};
                border-left: 3px solid {color};
                border-radius: 4px;
            }}
        """)
        check_label.setTextFormat(Qt.TextFormat.RichText)
    
    def update_header(self, text: str):
        """Update the header text."""
        self.header_label.setText(text)
    
    def clear(self):
        """Clear all validation checks."""
        # Remove all check labels
        for label in self.check_labels.values():
            self.checks_layout.removeWidget(label)
            label.deleteLater()
        
        self.checks.clear()
        self.check_labels.clear()
    
    def has_failed_checks(self) -> bool:
        """Check if any validation checks have failed."""
        return any(
            check.get('status') == self.STATUS_FAILED
            for check in self.checks.values()
        )
    
    def has_warnings(self) -> bool:
        """Check if any validation checks have warnings."""
        return any(
            check.get('status') == self.STATUS_WARNING
            for check in self.checks.values()
        )
    
    def all_passed(self) -> bool:
        """Check if all validation checks have passed."""
        if not self.checks:
            return False
        # Exclude "Status" check from validation
        relevant_checks = {k: v for k, v in self.checks.items() if k != "Status"}
        if not relevant_checks:
            return False
        return all(
            check.get('status') == self.STATUS_PASSED
            for check in relevant_checks.values()
        ) and not self.has_failed_checks()

