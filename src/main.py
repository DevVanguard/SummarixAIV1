"""Main entry point for SummarixAI application."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


def main():
    """Main application entry point."""
    # Ensure directories exist
    Config.ensure_directories()
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName("SummarixAI")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    logger.info(f"{Config.APP_NAME} started")
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

