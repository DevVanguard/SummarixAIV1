"""Main entry point for SummarixAI application."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import only lightweight modules first
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

# Import config and logger (lightweight)
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


def main():
    """Main application entry point."""
    # Ensure directories exist
    Config.ensure_directories()
    
    # Create Qt application FIRST (before heavy imports)
    app = QApplication(sys.argv)
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName("SummarixAI")
    
    # Show splash screen IMMEDIATELY (before any heavy imports)
    from src.gui.splash_screen import SplashScreen
    
    splash = SplashScreen()
    splash.show()
    
    # Process events to ensure splash is visible
    app.processEvents()
    
    # Store reference to main window
    main_window = None
    
    # Import and load main window after splash is shown
    def load_main_window():
        """Load main window after dependencies are ready."""
        nonlocal main_window
        try:
            logger.info("Loading main window...")
            # Update splash to show we're loading main window
            splash.update_progress(95, "Loading application interface...")
            app.processEvents()
            
            # Import main window (this will import heavy modules)
            logger.info("Importing MainWindow...")
            from src.gui.main_window import MainWindow
            
            # Create and show main window
            logger.info("Creating MainWindow instance...")
            main_window = MainWindow()
            logger.info("Showing MainWindow...")
            main_window.show()
            
            # Close splash screen
            logger.info("Closing splash screen...")
            splash.close()
            
            logger.info(f"{Config.APP_NAME} started successfully")
        except Exception as e:
            logger.error(f"Error loading main window: {e}")
            import traceback
            traceback.print_exc()
            splash.close()
            # Show error message
            try:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(None, "Error", f"Failed to start application:\n{str(e)}")
            except:
                pass
    
    # Start loading in background (this creates the loader)
    splash.start_loading()
    
    # Connect finished signal to load main window (AFTER loader is created)
    if splash.loader:
        splash.loader.finished.connect(load_main_window)
        logger.info("Connected finished signal to load_main_window")
    else:
        logger.error("Loader not created!")
    
    # Also add a timeout fallback in case signal doesn't fire
    def timeout_fallback():
        """Fallback if loader doesn't finish."""
        if splash.loader:
            if not splash.loader.is_loaded() and splash.loader.isRunning():
                logger.warning("Loader timeout - loading main window anyway")
                load_main_window()
        else:
            logger.warning("No loader - loading main window immediately")
            load_main_window()
    
    QTimer.singleShot(10000, timeout_fallback)  # 10 second timeout fallback
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

