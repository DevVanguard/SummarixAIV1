"""
Modern theme definitions for SummarixAI GUI.
This file contains all the styling for both dark and light themes.
Enhanced with professional design system following HCI principles:
- Clear visual hierarchy with depth perception
- Consistent spacing and alignment (8px grid system)
- WCAG AAA accessible color contrasts
- Modern card-based layouts with elevation
- Smooth micro-interactions and hover states
- Beautiful gradient accents and shadows
"""

DARK_THEME = """
/* Main window with sophisticated atmospheric gradient background */
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0a0e27, stop:0.3 #161b33, stop:0.7 #1a1f3a, stop:1 #0f1428);
}

/* Base widget styling - modern dark theme with enhanced readability */
QWidget {
    background-color: transparent;
    color: #e8eaed;
    font-family: 'Segoe UI', 'SF Pro Display', 'Inter', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    letter-spacing: 0.01em;
}

/* Modern button styling with sophisticated depth and micro-interactions */
/* Primary action buttons (vibrant blue-purple gradient) - enhanced with better visual feedback */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #667eea, stop:0.5 #5568d3, stop:1 #4c52cc);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    font-weight: 600;
    font-size: 11pt;
    min-height: 44px;
    letter-spacing: 0.6px;
}

/* Hover state - elevated with luminous glow effect */
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #7c8ff5, stop:0.5 #667eea, stop:1 #5568d3);
}

/* Pressed state - darker with inset effect simulating button press */
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #4c52cc, stop:1 #3d42b8);
}

/* Focus state - clear ring for accessibility (WCAG 2.1) */
QPushButton:focus {
    outline: 3px solid rgba(124, 143, 245, 0.6);
    outline-offset: 3px;
}

/* Disabled state - muted with clear visual indication */
QPushButton:disabled {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3a3a42, stop:1 #2d2d35);
    color: #737380;
    border: none;
    opacity: 0.5;
}

/* Input fields with smooth transitions and clear focus states */
QLineEdit, QTextEdit {
    background-color: #1c1e26;
    border: 2px solid #3a3d4a;
    border-radius: 12px;
    padding: 14px 18px;
    color: #e8eaed;
    selection-background-color: #667eea;
    selection-color: #ffffff;
    font-size: 10.5pt;
    line-height: 1.7;
}

QLineEdit:focus, QTextEdit:focus {
    border: 2px solid #7c8ff5;
    background-color: #22242e;
    outline: 2px solid rgba(124, 143, 245, 0.4);
    outline-offset: 2px;
}

QTextEdit {
    selection-background-color: #667eea;
    selection-color: #ffffff;
}

/* Radio buttons with modern design and clear selection states */
QRadioButton {
    spacing: 12px;
    color: #e8eaed;
    font-size: 10.5pt;
    padding: 6px 0px;
}

QRadioButton::indicator {
    width: 22px;
    height: 22px;
    border-radius: 11px;
    border: 2px solid #4a4d5a;
    background-color: #1c1e26;
}

QRadioButton::indicator:hover {
    border-color: #7c8ff5;
    background-color: #22242e;
}

QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
        stop:0 #ffffff, stop:0.4 #ffffff, stop:0.5 #667eea, stop:1 #667eea);
    border-color: #667eea;
}

QRadioButton::indicator:focus {
    outline: 2px solid rgba(124, 143, 245, 0.6);
    outline-offset: 3px;
}

/* Modern progress bar with smooth animated gradient and shimmer effect */
QProgressBar {
    border: none;
    border-radius: 12px;
    text-align: center;
    background-color: #1c1e26;
    color: #e8eaed;
    font-weight: 600;
    font-size: 9pt;
    height: 24px;
    padding: 2px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #667eea, stop:0.3 #7c8ff5, stop:0.7 #5568d3, stop:1 #764ba2);
    border-radius: 10px;
}

/* Enhanced labels with better readability */
QLabel {
    color: #e8eaed;
}

/* Modern menu bar with refined depth */
QMenuBar {
    background-color: #181a24;
    color: #e8eaed;
    border-bottom: 1px solid #2f3241;
    padding: 6px 10px;
    font-size: 10pt;
}

QMenuBar::item {
    background-color: transparent;
    padding: 8px 14px;
    border-radius: 8px;
    margin: 0px 2px;
}

QMenuBar::item:selected {
    background-color: #2a2c38;
}

QMenuBar::item:pressed {
    background-color: #667eea;
    color: #ffffff;
}

/* Modern dropdown menu with elevation shadow */
QMenu {
    background-color: #1f212b;
    color: #e8eaed;
    border: 1px solid #3a3d4a;
    border-radius: 10px;
    padding: 8px;
}

QMenu::item {
    padding: 10px 24px;
    border-radius: 7px;
}

QMenu::item:selected {
    background-color: #667eea;
    color: #ffffff;
}

/* Enhanced status bar with vibrant gradient */
QStatusBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #667eea, stop:0.5 #5568d3, stop:1 #764ba2);
    color: #ffffff;
    font-weight: 500;
    border: none;
    padding: 6px 14px;
    font-size: 9.5pt;
}

/* Modern scrollbar with smooth hover effects and subtle design */
QScrollBar:vertical {
    background-color: #181a24;
    width: 16px;
    border: none;
    border-radius: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3a3d4a, stop:1 #4a4d5a);
    min-height: 40px;
    border-radius: 8px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4a4d5a, stop:1 #5a5d6a);
}

QScrollBar::handle:vertical:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #667eea, stop:1 #7c8ff5);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
    border: none;
}

/* Group boxes with modern card design and sophisticated elevation */
QGroupBox {
    border: 1px solid #2f3241;
    border-radius: 16px;
    margin-top: 18px;
    padding-top: 20px;
    padding-bottom: 16px;
    padding-left: 16px;
    padding-right: 16px;
    font-weight: 600;
    font-size: 11pt;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1f212b, stop:1 #1a1c26);
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 20px;
    top: 6px;
    padding: 0px 12px;
    color: #7c8ff5;
    font-weight: 700;
    font-size: 11pt;
    letter-spacing: 0.8px;
    background-color: transparent;
}

/* Modern combo box with smooth interactions and elevation */
QComboBox {
    background-color: #1c1e26;
    border: 2px solid #3a3d4a;
    border-radius: 10px;
    padding: 10px 16px;
    padding-right: 34px;
    min-width: 150px;
    font-size: 10pt;
    color: #e8eaed;
}

QComboBox:hover {
    border-color: #7c8ff5;
    background-color: #22242e;
}

QComboBox:focus {
    border-color: #7c8ff5;
    background-color: #22242e;
    outline: 2px solid rgba(124, 143, 245, 0.4);
    outline-offset: 2px;
}

/* Disabled state */
QComboBox:disabled {
    background-color: #181a24;
    color: #5a5d6a;
    border-color: #2a2c38;
    opacity: 0.5;
}

/* Drop-down arrow area */
QComboBox::drop-down {
    border: none;
    width: 34px;
    padding-right: 10px;
}

/* Custom arrow with better visual */
QComboBox::down-arrow {
    image: none;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid #e8eaed;
    width: 0;
    height: 0;
}

QComboBox::down-arrow:hover {
    border-top: 8px solid #7c8ff5;
}

/* Dropdown list with elevation */
QComboBox QAbstractItemView {
    background-color: #1f212b;
    border: 1px solid #3a3d4a;
    border-radius: 10px;
    padding: 6px;
    selection-background-color: #667eea;
    selection-color: #ffffff;
    outline: none;
}

QComboBox QAbstractItemView::item {
    min-height: 36px;
    padding: 6px 14px;
    border-radius: 7px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #2a2c38;
}

QComboBox QAbstractItemView::item:selected {
    background-color: #667eea;
    color: #ffffff;
}

/* Semantic color classes for buttons - can be applied via setProperty */
/* Success button - vibrant emerald green with modern gradient */
QPushButton[buttonType="success"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #48bb78, stop:0.5 #38a169, stop:1 #2f855a);
    border: none;
}

QPushButton[buttonType="success"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #68d391, stop:0.5 #48bb78, stop:1 #38a169);
}

QPushButton[buttonType="success"]:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2f855a, stop:1 #276749);
}

/* Danger button - vibrant red-rose with modern gradient */
QPushButton[buttonType="danger"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fc8181, stop:0.5 #f56565, stop:1 #e53e3e);
    border: none;
}

QPushButton[buttonType="danger"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #feb2b2, stop:0.5 #fc8181, stop:1 #f56565);
}

QPushButton[buttonType="danger"]:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #e53e3e, stop:1 #c53030);
}

/* Professional exit button - circular with modern styling */
QPushButton[buttonType="exit"] {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
        stop:0 #3a3d4a, stop:1 #2a2c38);
    color: #e8eaed;
    border: 1px solid #4a4d5a;
    border-radius: 18px;
    font-size: 16pt;
    font-weight: 300;
    padding: 0px;
    min-width: 36px;
    max-width: 36px;
    min-height: 36px;
    max-height: 36px;
}

QPushButton[buttonType="exit"]:hover {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
        stop:0 #fc8181, stop:0.5 #f56565, stop:1 #e53e3e);
    color: #ffffff;
    border: 1px solid #fc8181;
}

QPushButton[buttonType="exit"]:pressed {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
        stop:0 #e53e3e, stop:1 #c53030);
    border: 1px solid #c53030;
}

QPushButton[buttonType="exit"]:focus {
    outline: 2px solid rgba(124, 143, 245, 0.6);
    outline-offset: 2px;
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

/* Professional exit button - circular with subtle styling */
QPushButton[buttonType="exit"] {
    background-color: #e0e0e0;
    color: #1e1e1e;
    border: 1px solid #d0d0d0;
    border-radius: 14px;
    font-size: 16pt;
    font-weight: 300;
    padding: 0px;
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
}

QPushButton[buttonType="exit"]:hover {
    background-color: #d0d0d0;
    border-color: #b0b0b0;
    color: #000000;
}

QPushButton[buttonType="exit"]:pressed {
    background-color: #c0c0c0;
    border-color: #a0a0a0;
}

QPushButton[buttonType="exit"]:focus {
    outline: 2px solid #0078d4;
    outline-offset: 2px;
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

