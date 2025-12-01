"""Splash screen with progress indicator for application startup."""

import logging
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from src.utils.config import Config

logger = logging.getLogger(__name__)


class StartupLoader(QThread):
    """Thread to load heavy dependencies in background."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self):
        """Initialize startup loader."""
        super().__init__()
        self._loaded = False
    
    def run(self):
        """Load dependencies."""
        try:
            # Step 1: Load PyQt6 (already loaded, but check)
            self.progress.emit(10, "Initializing GUI framework...")
            self.msleep(100)  # Small delay for smooth progress
            
            # Step 2: Load PyTorch
            self.progress.emit(20, "Loading PyTorch...")
            try:
                import torch
                logger.info("PyTorch loaded")
            except Exception as e:
                logger.warning(f"PyTorch loading warning: {e}")
            self.msleep(200)
            
            # Step 3: Load Transformers
            self.progress.emit(40, "Loading Transformers library...")
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                logger.info("Transformers loaded")
            except Exception as e:
                logger.warning(f"Transformers loading warning: {e}")
            self.msleep(200)
            
            # Step 4: Load other dependencies
            self.progress.emit(60, "Loading dependencies...")
            try:
                import networkx
                import nltk
                import sklearn
                logger.info("Dependencies loaded")
            except Exception as e:
                logger.warning(f"Dependencies loading warning: {e}")
            self.msleep(200)
            
            # Step 5: Initialize core modules (lightweight)
            self.progress.emit(80, "Initializing core modules...")
            try:
                # Import lightweight modules only
                from src.core.pdf_processor import PDFProcessor
                logger.info("PDF processor initialized")
            except Exception as e:
                logger.error(f"Core modules error: {e}")
            self.msleep(200)
            
            # Step 6: Check models
            self.progress.emit(90, "Checking model files...")
            model_path = Config.get_model_path()
            if model_path.exists():
                logger.info(f"Models found at: {model_path}")
            else:
                logger.warning(f"Models not found at: {model_path}")
            self.msleep(200)
            
            # Complete
            self.progress.emit(100, "Ready!")
            self.msleep(300)
            
            self._loaded = True
            self.finished.emit()
            
        except Exception as e:
            logger.error(f"Startup loader error: {str(e)}")
            self.error.emit(str(e))
    
    def is_loaded(self) -> bool:
        """Check if loading is complete."""
        return self._loaded


class SplashScreen(QWidget):
    """Custom splash screen with progress bar."""
    
    def __init__(self):
        """Initialize splash screen."""
        super().__init__()
        
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SplashScreen |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        
        self.setFixedSize(600, 400)
        self._setup_ui()
        self.loader: Optional[StartupLoader] = None
        
        # Center on screen
        self._center_on_screen()
    
    def _setup_ui(self):
        """Set up splash screen UI."""
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # App name
        title_label = QLabel(Config.APP_NAME)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #0078d4;")
        
        # Version
        version_label = QLabel(f"Version {Config.APP_VERSION}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #808080; font-size: 12pt;")
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #e0e0e0; font-size: 11pt; margin-top: 20px;")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
                color: #e0e0e0;
                height: 25px;
                font-size: 10pt;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        
        # Layout
        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(version_label)
        layout.addStretch()
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #1e1e1e;")
    
    def _center_on_screen(self):
        """Center splash screen on screen."""
        from PyQt6.QtGui import QScreen
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            self.move(x, y)
    
    def start_loading(self):
        """Start loading process."""
        logger.info("Starting loader thread...")
        self.loader = StartupLoader()
        self.loader.progress.connect(self.update_progress)
        self.loader.finished.connect(self.on_loading_finished)
        self.loader.error.connect(self.on_loading_error)
        logger.info("Starting loader thread...")
        self.loader.start()
        logger.info("Loader thread started")
    
    def update_progress(self, value: int, message: str):
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        QApplication.processEvents()  # Keep UI responsive
    
    def on_loading_finished(self):
        """Handle loading completion."""
        self.status_label.setText("Ready!")
        self.progress_bar.setValue(100)
        QApplication.processEvents()
        # Note: Main window loading is handled by the signal connection in main.py
    
    def on_loading_error(self, error: str):
        """Handle loading error."""
        self.status_label.setText(f"Error: {error}")
        logger.error(f"Splash screen loading error: {error}")
    
    def finish(self, widget):
        """Close splash screen (compatible with QSplashScreen API)."""
        self.close()

