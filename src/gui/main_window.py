"""Main application window for SummarixAI."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.abstractive.summarizer import AbstractiveSummarizer
from src.core.extractive.textrank import TextRankSummarizer
from src.core.pdf_processor import PDFProcessor
from src.gui.components.file_upload import FileUploadWidget
from src.gui.components.mode_selector import ModeSelectorWidget, SummarizationMode
from src.gui.components.progress_indicator import ProgressIndicatorWidget
from src.gui.components.summary_display import SummaryDisplayWidget
from src.gui.styles.theme import get_theme
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


class SummarizationWorker(QThread):
    """Worker thread for summarization processing."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(
        self,
        file_path: Path,
        mode: SummarizationMode,
        extractive_summarizer: TextRankSummarizer,
        abstractive_summarizer: Optional[AbstractiveSummarizer] = None
    ):
        """Initialize worker thread."""
        super().__init__()
        self.file_path = file_path
        self.mode = mode
        self.extractive_summarizer = extractive_summarizer
        self.abstractive_summarizer = abstractive_summarizer
    
    def run(self):
        """Run the summarization process."""
        try:
            self.progress.emit(10, "Loading PDF...")
            
            # Process PDF
            with PDFProcessor() as processor:
                if not processor.load_pdf(self.file_path):
                    self.error.emit("Failed to load PDF file")
                    return
                
                self.progress.emit(30, "Extracting text from PDF...")
                text = processor.extract_text()
                
                if not text or not text.strip():
                    self.error.emit("No text found in PDF")
                    return
                
                self.progress.emit(50, "Generating summary...")
                
                # Generate summary based on mode
                if self.mode == SummarizationMode.EXTRACTIVE:
                    summary = self.extractive_summarizer.summarize(text)
                else:
                    # Abstractive mode
                    if not self.abstractive_summarizer:
                        self.error.emit("Abstractive summarizer not initialized")
                        return
                    
                    if not self.abstractive_summarizer.is_ready():
                        self.progress.emit(60, "Loading model...")
                        if not self.abstractive_summarizer.initialize():
                            self.error.emit("Failed to load abstractive model")
                            return
                    
                    self.progress.emit(70, "Generating abstractive summary...")
                    summary = self.abstractive_summarizer.summarize(text)
                
                self.progress.emit(100, "Summary complete!")
                self.finished.emit(summary)
                
        except Exception as e:
            logger.error(f"Error in summarization worker: {str(e)}")
            self.error.emit(f"Error: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        
        self.current_file: Optional[Path] = None
        self.worker: Optional[SummarizationWorker] = None
        
        # Initialize summarizers
        self.extractive_summarizer = TextRankSummarizer()
        self.abstractive_summarizer = AbstractiveSummarizer()
        
        self._setup_ui()
        self._setup_menu()
        self._apply_theme()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle(f"{Config.APP_NAME} v{Config.APP_VERSION}")
        self.setMinimumSize(900, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # File upload
        self.file_upload = FileUploadWidget()
        self.file_upload.file_selected.connect(self._on_file_selected)
        layout.addWidget(self.file_upload)
        
        # Mode selector
        self.mode_selector = ModeSelectorWidget()
        layout.addWidget(self.mode_selector)
        
        # Summarize button
        self.summarize_button = QPushButton("Summarize")
        self.summarize_button.setEnabled(False)
        self.summarize_button.clicked.connect(self._on_summarize)
        self.summarize_button.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                font-weight: 600;
                padding: 12px;
                min-height: 40px;
            }
        """)
        layout.addWidget(self.summarize_button)
        
        # Progress indicator
        self.progress_indicator = ProgressIndicatorWidget()
        layout.addWidget(self.progress_indicator)
        
        # Summary display
        self.summary_display = SummaryDisplayWidget()
        layout.addWidget(self.summary_display)
        
        central_widget.setLayout(layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def _setup_menu(self):
        """Set up menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open PDF...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _apply_theme(self):
        """Apply dark theme."""
        self.setStyleSheet(get_theme("dark"))
    
    def _on_file_selected(self, file_path: Path):
        """Handle file selection."""
        self.current_file = file_path
        self.summarize_button.setEnabled(True)
        self.statusBar().showMessage(f"File selected: {file_path.name}")
    
    def _on_open_file(self):
        """Handle open file menu action."""
        self.file_upload._browse_files()
    
    def _on_summarize(self):
        """Handle summarize button click."""
        if not self.current_file:
            QMessageBox.warning(self, "No File", "Please select a PDF file first.")
            return
        
        # Disable button during processing
        self.summarize_button.setEnabled(False)
        self.summary_display.clear_summary()
        
        # Get selected mode
        mode = self.mode_selector.get_mode()
        
        # Show progress
        mode_str = "extractive" if mode == SummarizationMode.EXTRACTIVE else "abstractive"
        self.progress_indicator.show_progress(f"Starting {mode_str} summarization...")
        
        # Create and start worker thread
        self.worker = SummarizationWorker(
            self.current_file,
            mode,
            self.extractive_summarizer,
            self.abstractive_summarizer if mode == SummarizationMode.ABSTRACTIVE else None
        )
        self.worker.progress.connect(self._on_progress_update)
        self.worker.finished.connect(self._on_summarization_finished)
        self.worker.error.connect(self._on_summarization_error)
        self.worker.start()
    
    def _on_progress_update(self, value: int, message: str):
        """Handle progress updates."""
        self.progress_indicator.update_progress(value, message)
        self.statusBar().showMessage(message)
    
    def _on_summarization_finished(self, summary: str):
        """Handle summarization completion."""
        self.progress_indicator.hide_progress()
        self.summary_display.set_summary(summary)
        self.summarize_button.setEnabled(True)
        self.statusBar().showMessage("Summary generated successfully")
        
        # Clean up worker
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None
    
    def _on_summarization_error(self, error_message: str):
        """Handle summarization error."""
        self.progress_indicator.hide_progress()
        self.summarize_button.setEnabled(True)
        self.statusBar().showMessage("Error occurred")
        
        QMessageBox.critical(self, "Error", error_message)
        
        # Clean up worker
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            f"About {Config.APP_NAME}",
            f"""
            <h2>{Config.APP_NAME} v{Config.APP_VERSION}</h2>
            <p>A standalone, offline-capable desktop application for document summarization.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Extractive summarization (TextRank)</li>
                <li>Abstractive summarization (T5-small)</li>
                <li>Fully offline operation</li>
                <li>Privacy-first design</li>
            </ul>
            <p>All processing occurs locally on your device.</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop worker if running
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        # Clean up abstractive summarizer
        if self.abstractive_summarizer:
            self.abstractive_summarizer.cleanup()
        
        event.accept()

