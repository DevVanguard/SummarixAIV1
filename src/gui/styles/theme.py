"""
Modern theme definitions for SummarixAI GUI.
This file contains all the styling for both dark and light themes.
We use a modern, professional design with smooth gradients and better visual hierarchy.
"""

DARK_THEME = """
/* Main window with subtle gradient background */
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1a1a1a, stop:1 #1e1e1e);
}

/* Base widget styling - modern dark theme */
QWidget {
    background-color: transparent;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    font-size: 10pt;
}

/* Modern button styling with gradient, shadows (simulated), and hover effects */
/* Primary action buttons (blue) - default style */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0078d4, stop:1 #005a9e);
    color: white;
    border: 1px solid #004578;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 10pt;
    min-height: 32px;
}

/* Hover state - lighter blue with elevation effect (simulated shadow) */
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #106ebe, stop:1 #0078d4);
    border: 1px solid #005a9e;
    border-top: 2px solid #40a8ff;
}

/* Pressed state - darker blue with pressed effect */
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #005a9e, stop:1 #004578);
    border: 1px solid #003d5c;
    border-top: 1px solid #004578;
}

/* Focus state - visible outline for keyboard navigation */
QPushButton:focus {
    outline: 2px solid #40a8ff;
    outline-offset: 2px;
}

/* Disabled state - grayed out with reduced opacity effect */
QPushButton:disabled {
    background-color: #3d3d3d;
    color: #6c757d;
    border: 1px solid #2d2d2d;
    opacity: 0.6;
}

/* Input fields with better focus states */
QLineEdit, QTextEdit {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e0e0e0;
    selection-background-color: #0078d4;
    selection-color: white;
}

QLineEdit:focus, QTextEdit:focus {
    border: 2px solid #0078d4;
    background-color: #252525;
    outline: 1px solid #40a8ff;
    outline-offset: 1px;
}

QTextEdit {
    selection-background-color: #0078d4;
}

QRadioButton {
    spacing: 8px;
    color: #e0e0e0;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #3d3d3d;
    background-color: #2d2d2d;
}

QRadioButton::indicator:hover {
    border-color: #0078d4;
}

QRadioButton::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}

QRadioButton::indicator:focus {
    border: 2px solid #40a8ff;
    outline: 1px solid #40a8ff;
    outline-offset: 1px;
}

/* Modern progress bar with gradient */
QProgressBar {
    border: 2px solid #3d3d3d;
    border-radius: 8px;
    text-align: center;
    background-color: #2d2d2d;
    color: #e0e0e0;
    font-weight: 500;
    height: 24px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0078d4, stop:1 #00a8ff);
    border-radius: 6px;
}

QLabel {
    color: #e0e0e0;
}

QMenuBar {
    background-color: #252526;
    color: #e0e0e0;
    border-bottom: 1px solid #3d3d3d;
}

QMenuBar::item {
    background-color: transparent;
    padding: 4px 8px;
}

QMenuBar::item:selected {
    background-color: #2d2d2d;
}

QMenu {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
}

QMenu::item:selected {
    background-color: #0078d4;
}

QStatusBar {
    background-color: #0078d4;
    color: white;
    border-top: 1px solid #005a9e;
}

QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #3d3d3d;
    min-height: 20px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4d4d4d;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Group boxes with card-like appearance */
QGroupBox {
    border: 2px solid #3d3d3d;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: 600;
    font-size: 11pt;
    background-color: #252525;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0px 8px;
    color: #4ec9b0;
}

/* Card-like depth effect for group boxes */
QGroupBox {
    border: 2px solid #3d3d3d;
    border-top: 2px solid #4d4d4d;
    border-left: 2px solid #4d4d4d;
    border-bottom: 2px solid #2d2d2d;
    border-right: 2px solid #2d2d2d;
}

/* Combo box styling */
QComboBox {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #0078d4;
}

QComboBox:focus {
    border-color: #0078d4;
    background-color: #252525;
    outline: 1px solid #40a8ff;
    outline-offset: 1px;
}

/* Disabled combo box styling */
QComboBox:disabled {
    background-color: #2d2d2d;
    color: #6c757d;
    border-color: #2d2d2d;
    opacity: 0.6;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #e0e0e0;
    width: 0;
    height: 0;
}

/* Semantic color classes for buttons - can be applied via setProperty */
QPushButton[buttonType="success"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #28a745, stop:1 #1e7e34);
    border: 1px solid #155724;
}

QPushButton[buttonType="success"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #34ce57, stop:1 #28a745);
    border: 1px solid #1e7e34;
    border-top: 2px solid #5ade7a;
}

QPushButton[buttonType="success"]:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1e7e34, stop:1 #155724);
    border: 1px solid #0f4a1a;
}

QPushButton[buttonType="danger"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #dc3545, stop:1 #c82333);
    border: 1px solid #bd2130;
}

QPushButton[buttonType="danger"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #e4606d, stop:1 #dc3545);
    border: 1px solid #c82333;
    border-top: 2px solid #ff6b7a;
}

QPushButton[buttonType="danger"]:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #c82333, stop:1 #bd2130);
    border: 1px solid #a71e2a;
}
"""

LIGHT_THEME = """
QMainWindow {
    background-color: #ffffff;
}

QWidget {
    background-color: #ffffff;
    color: #1e1e1e;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}

QPushButton {
    background-color: #0078d4;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 32px;
}

QPushButton:hover {
    background-color: #106ebe;
}

QPushButton:pressed {
    background-color: #005a9e;
}

QPushButton:disabled {
    background-color: #e0e0e0;
    color: #808080;
}

QLineEdit, QTextEdit {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 6px;
    color: #1e1e1e;
}

QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #0078d4;
}

QTextEdit {
    selection-background-color: #0078d4;
    selection-color: white;
}

QRadioButton {
    spacing: 8px;
    color: #1e1e1e;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #d0d0d0;
    background-color: #ffffff;
}

QRadioButton::indicator:hover {
    border-color: #0078d4;
}

QRadioButton::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}

QProgressBar {
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    text-align: center;
    background-color: #f5f5f5;
    color: #1e1e1e;
}

QProgressBar::chunk {
    background-color: #0078d4;
    border-radius: 3px;
}

QLabel {
    color: #1e1e1e;
}

QMenuBar {
    background-color: #f3f3f3;
    color: #1e1e1e;
    border-bottom: 1px solid #d0d0d0;
}

QMenuBar::item {
    background-color: transparent;
    padding: 4px 8px;
}

QMenuBar::item:selected {
    background-color: #e0e0e0;
}

QMenu {
    background-color: #ffffff;
    color: #1e1e1e;
    border: 1px solid #d0d0d0;
}

QMenu::item:selected {
    background-color: #0078d4;
    color: white;
}

QStatusBar {
    background-color: #0078d4;
    color: white;
    border-top: 1px solid #005a9e;
}

QScrollBar:vertical {
    background-color: #f5f5f5;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #d0d0d0;
    min-height: 20px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background-color: #b0b0b0;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QGroupBox {
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 12px;
    font-weight: 500;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0px 6px;
}
"""


def get_theme(theme_name: str = "dark") -> str:
    """
    Get theme stylesheet.
    
    Args:
        theme_name: 'dark' or 'light'
        
    Returns:
        QSS stylesheet string
    """
    if theme_name.lower() == "light":
        return LIGHT_THEME
    return DARK_THEME

