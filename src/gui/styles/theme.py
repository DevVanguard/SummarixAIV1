"""Modern theme definitions for SummarixAI GUI."""

DARK_THEME = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
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
    background-color: #3d3d3d;
    color: #808080;
}

QLineEdit, QTextEdit {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 6px;
    color: #e0e0e0;
}

QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #0078d4;
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

QProgressBar {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    text-align: center;
    background-color: #2d2d2d;
    color: #e0e0e0;
}

QProgressBar::chunk {
    background-color: #0078d4;
    border-radius: 3px;
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

QGroupBox {
    border: 1px solid #3d3d3d;
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

