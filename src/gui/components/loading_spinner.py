"""Sleek circular loading spinner component."""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class LoadingSpinner(QWidget):
    """
    Sleek circular loading spinner animation.
    Displays a rotating circle animation for processing states.
    """
    
    def __init__(self, parent=None, size: int = 48, line_width: int = 4):
        """
        Initialize loading spinner.
        
        Args:
            parent: Parent widget
            size: Size of the spinner in pixels (default: 48)
            line_width: Width of the spinner lines (default: 4)
        """
        super().__init__(parent)
        self.size = size
        self.line_width = line_width
        self.angle = 0
        self.timer: Optional[QTimer] = None
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.hide()
    
    def start(self):
        """Start the spinner animation."""
        if self.timer is None:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._update_animation)
            self.timer.start(16)  # ~60 FPS for smooth animation
        self.show()
    
    def stop(self):
        """Stop the spinner animation."""
        if self.timer:
            self.timer.stop()
            self.timer = None
        self.hide()
    
    def _update_animation(self):
        """Update animation frame."""
        self.angle = (self.angle + 8) % 360
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the spinner animation."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate center and radius
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(self.width(), self.height()) / 2 - self.line_width
        
        # Draw rotating arc
        rect = QRectF(
            center_x - radius,
            center_y - radius,
            radius * 2,
            radius * 2
        )
        
        # Create gradient-like effect with multiple arcs (vibrant blue-purple)
        for i in range(8):
            # Calculate angle for this segment
            segment_angle = (self.angle + i * 45) % 360
            
            # Calculate opacity (fade effect)
            opacity = 1.0 - (i / 8.0) * 0.7
            opacity = max(0.1, opacity)
            
            # Create pen with gradient opacity - using new color scheme
            pen = QPen(QColor(124, 143, 245, int(255 * opacity)))  # #7c8ff5 with opacity
            pen.setWidth(self.line_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            
            # Draw arc segment
            start_angle = segment_angle * 16  # Qt uses 1/16th degree units
            span_angle = 30 * 16  # 30 degree span
            
            painter.drawArc(rect, int(start_angle), int(span_angle))

