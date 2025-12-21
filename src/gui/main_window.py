"""
Main application window for SummarixAI.
This is the central hub of the application - it coordinates all the UI components
and handles the summarization workflow from file selection to result display.
"""

import logging
from pathlib import Path
from typing import Optional

import time
from PyQt6.QtCore import QThread, QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from src.core.abstractive.summarizer import AbstractiveSummarizer
from src.core.extractive.textrank import TextRankSummarizer
from src.core.pdf_processor import PDFProcessor
from src.gui.components.file_upload import FileUploadWidget
from src.gui.components.loading_spinner import LoadingSpinner
from src.gui.components.mode_selector import ModeSelectorWidget, SummarizationMode
from src.gui.components.progress_indicator import ProgressIndicatorWidget
from src.gui.components.summary_display import SummaryDisplayWidget
from src.gui.styles.theme import get_theme
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


class SummarizationWorker(QThread):
    """
    Worker thread for summarization processing.
    This runs in a separate thread so the UI stays responsive during long operations.
    We use signals to communicate progress and results back to the main thread.
    """
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(
        self,
        file_path: Path,
        mode: SummarizationMode,
        extractive_summarizer: TextRankSummarizer,
        abstractive_summarizer: Optional[AbstractiveSummarizer] = None,
        mode_selector = None
    ):
        """
        Initialize worker thread with all necessary components.
        
        Args:
            file_path: Path to the PDF file to summarize
            mode: Which summarization mode to use (extractive or abstractive)
            extractive_summarizer: The extractive summarizer instance
            abstractive_summarizer: The abstractive summarizer instance (optional)
            mode_selector: Reference to mode selector widget to get length preset
        """
        super().__init__()
        self.file_path = file_path
        self.mode = mode
        self.extractive_summarizer = extractive_summarizer
        self.abstractive_summarizer = abstractive_summarizer
        self.mode_selector = mode_selector
    
    def run(self):
        """
        Main worker thread execution.
        This method runs in a separate thread and performs all the heavy lifting.
        We emit signals to update the UI thread with progress and results.
        """
        try:
            self.progress.emit(10, "Loading PDF file...")
            
            # Initialize text variable outside try block so it's accessible for summarization
            text = None
            
            # Open and process the PDF file
            # Using context manager ensures proper cleanup
            try:
                with PDFProcessor() as processor:
                    if not processor.load_pdf(self.file_path):
                        self.error.emit(
                            f"Failed to load PDF file.\n\n"
                            f"File: {self.file_path.name}\n"
                            f"Please ensure the file is a valid PDF and not corrupted."
                        )
                        return
                    
                    self.progress.emit(30, "Extracting text from PDF...")
                    
                    # Extract all text from the PDF
                    try:
                        text = processor.extract_text()
                    except Exception as e:
                        logger.error(f"Text extraction failed: {str(e)}", exc_info=True)
                        self.error.emit(
                            f"Failed to extract text from PDF:\n{str(e)}\n\n"
                            f"The PDF might be image-based or corrupted."
                        )
                        return
                    
                    # Validate that we got some text
                    if not text or not text.strip():
                        self.error.emit(
                            "No text found in PDF.\n\n"
                            "This PDF might be:\n"
                            "- Image-based (scanned document)\n"
                            "- Empty or corrupted\n"
                            "- Password protected"
                        )
                        return
                    
                    # Check text length - warn if very short
                    text_length = len(text.strip())
                    if text_length < 100:
                        logger.warning(f"Very short text extracted: {text_length} characters")
                    
            except Exception as e:
                # Catch any errors during PDF processing
                logger.error(f"PDF processing error: {str(e)}", exc_info=True)
                self.error.emit(f"Error processing PDF file:\n{str(e)}")
                return
            
            # If we got here, PDF processing was successful
            # Now proceed with summarization (this code is outside the exception handler)
            if text is None or not text.strip():
                self.error.emit("No text available for summarization")
                return
            
            self.progress.emit(50, "Generating summary...")
            
            # Generate summary based on selected mode
            if self.mode == SummarizationMode.EXTRACTIVE:
                # Extractive mode - fast, no model needed
                try:
                    # Quick check of document structure
                    sentences = self.extractive_summarizer.preprocessor.tokenize_sentences(text)
                    if not sentences:
                        self.error.emit("No sentences found in document")
                        return
                    
                    # Provide progress updates during processing
                    self.progress.emit(60, f"Analyzing {len(sentences)} sentences...")
                    
                    # The actual summarization can take time for large documents
                    # TF-IDF vectorization and PageRank computation happen here
                    summary = self.extractive_summarizer.summarize(text)
                    
                    if not summary or not summary.strip():
                        self.error.emit("Extractive summarization produced empty result")
                        return
                    
                    # Summary complete
                    self.progress.emit(95, "Summary generated successfully")
                    self.progress.emit(100, "Summary complete!")
                    self.finished.emit(summary)
                    
                except Exception as e:
                    logger.error(f"Extractive summarization failed: {str(e)}", exc_info=True)
                    self.error.emit(f"Extractive summarization error: {str(e)}")
                    return
            else:
                # Abstractive mode - requires model loading
                if not self.abstractive_summarizer:
                    self.error.emit("Abstractive summarizer not initialized")
                    return
                
                # Load model if not already loaded
                if not self.abstractive_summarizer.is_ready():
                    self.progress.emit(60, "Loading AI model (this may take a moment)...")
                    try:
                        if not self.abstractive_summarizer.initialize():
                            self.error.emit("Failed to load abstractive model. Please check if models are downloaded.")
                            return
                    except Exception as e:
                        logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
                        self.error.emit(f"Model loading error: {str(e)}")
                        return
                
                # Get length preset from mode selector if available
                max_output_tokens = Config.MAX_OUTPUT_TOKENS  # Default
                min_output_tokens = Config.MIN_SUMMARY_LENGTH  # Default
                length_penalty = Config.LENGTH_PENALTY  # Default
                
                if self.mode_selector:
                    try:
                        length_preset = self.mode_selector.get_length_preset()
                        max_output_tokens = Config.get_output_length_for_preset(length_preset)
                        min_output_tokens = Config.get_min_length_for_preset(length_preset)
                        length_penalty = Config.get_length_penalty_for_preset(length_preset)
                        logger.info(
                            f"Using length preset: {length_preset} "
                            f"(max: {max_output_tokens}, min: {min_output_tokens}, penalty: {length_penalty})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get length preset, using default: {str(e)}")
                        max_output_tokens = Config.MAX_OUTPUT_TOKENS
                        min_output_tokens = Config.MIN_SUMMARY_LENGTH
                        length_penalty = Config.LENGTH_PENALTY
                
                # Generate abstractive summary with specified length and preset
                try:
                    self.progress.emit(70, "Generating abstractive summary with AI...")
                    # Get preset for formatting
                    preset = "medium"  # Default
                    if self.mode_selector:
                        try:
                            preset = self.mode_selector.get_length_preset()
                        except Exception:
                            pass
                    
                    summary = self.abstractive_summarizer.summarize(
                        text,
                        max_length=max_output_tokens,
                        min_length=min_output_tokens,
                        length_penalty=length_penalty,
                        preset=preset
                    )
                    if not summary or not summary.strip():
                        self.error.emit("Abstractive summarization produced empty result")
                        return
                    
                    # If we got here, summarization was successful
                    self.progress.emit(100, "Summary complete!")
                    self.finished.emit(summary)
                    
                except Exception as e:
                    logger.error(f"Abstractive summarization failed: {str(e)}", exc_info=True)
                    self.error.emit(f"Summary generation error: {str(e)}")
                    return
                
        except Exception as e:
            # Catch-all for any unexpected errors
            logger.error(f"Unexpected error in summarization worker: {str(e)}", exc_info=True)
            self.error.emit(
                f"An unexpected error occurred:\n{str(e)}\n\n"
                f"Please check the logs for more details."
            )


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
        """
        Set up the user interface following HCI principles.
        Efficient space usage, clear visual hierarchy, proper spacing.
        """
        self.setWindowTitle(f"{Config.APP_NAME} v{Config.APP_VERSION}")
        self.setMinimumSize(900, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with enhanced professional spacing
        layout = QVBoxLayout()
        layout.setSpacing(16)  # Better spacing for visual breathing room
        layout.setContentsMargins(20, 16, 20, 16)  # Generous margins for modern look
        
        # File upload
        self.file_upload = FileUploadWidget()
        self.file_upload.file_selected.connect(self._on_file_selected)
        self.file_upload.file_cleared.connect(self._on_file_cleared)
        self.file_upload.validation_complete.connect(self._on_validation_complete)
        layout.addWidget(self.file_upload)
        
        # Mode selector
        self.mode_selector = ModeSelectorWidget()
        layout.addWidget(self.mode_selector)
        
        # Summarize button - enhanced professional styling with depth
        self.summarize_button = QPushButton("⚡ Summarize")
        self.summarize_button.setEnabled(False)  # Disabled until file is selected
        self.summarize_button.clicked.connect(self._on_summarize)
        self.summarize_button.setFixedHeight(44)  # Slightly taller for better presence
        self.summarize_button.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                font-weight: 700;
                padding: 12px 28px;
                border-radius: 8px;
                letter-spacing: 0.5px;
            }
            QPushButton:disabled {
                opacity: 0.5;
                background-color: #3d3d3d;
                color: #6c757d;
            }
        """)
        layout.addWidget(self.summarize_button)
        
        # Loading spinner - sleek circular animation below button
        spinner_container = QWidget()
        spinner_layout = QVBoxLayout()
        spinner_layout.setContentsMargins(0, 8, 0, 8)
        spinner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_spinner = LoadingSpinner(size=48, line_width=4)
        spinner_layout.addWidget(self.loading_spinner, alignment=Qt.AlignmentFlag.AlignCenter)
        spinner_container.setLayout(spinner_layout)
        layout.addWidget(spinner_container)
        
        # Progress indicator (kept for time/memory info, but hidden during processing)
        self.progress_indicator = ProgressIndicatorWidget()
        layout.addWidget(self.progress_indicator)
        
        # Summary display - make it expandable to use available space
        # Hidden initially, shown only when summary is generated
        self.summary_display = SummaryDisplayWidget()
        layout.addWidget(self.summary_display, 1)  # Give it stretch factor to expand
        self.summary_display.hide()  # Hide initially until summary is generated
        
        central_widget.setLayout(layout)
        
        # Set size policy for better space utilization
        self.summary_display.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Status bar with enhanced styling
        self.statusBar().showMessage("Ready")
        
        # Timer for periodic updates of time and memory during processing
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_progress_info)
        self.update_timer.setInterval(500)  # Update every 500ms
    
    def _setup_menu(self):
        """Set up menu bar with badges and exit button."""
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
        
        # Create right-side widget container for badges and exit button
        right_widget = QWidget()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(8, 4, 8, 4)
        right_layout.setSpacing(8)
        
        # Privacy badge - our biggest differentiator
        privacy_badge = QLabel("🔒 Privacy-First")
        privacy_badge.setToolTip("All processing occurs locally on your device. No data leaves your system.")
        privacy_badge.setStyleSheet("""
            QLabel {
                background-color: #28a745;
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 9pt;
                font-weight: 600;
            }
        """)
        
        # Offline assurance badge - our biggest differentiator
        offline_badge = QLabel("📡 Fully Offline")
        offline_badge.setToolTip("No internet connection required. All AI processing happens locally.")
        offline_badge.setStyleSheet("""
            QLabel {
                background-color: #0078d4;
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 9pt;
                font-weight: 600;
            }
        """)
        
        right_layout.addWidget(privacy_badge)
        right_layout.addWidget(offline_badge)
        
        right_widget.setLayout(right_layout)
        
        # Make sure the widget is visible and properly sized
        right_widget.setMinimumHeight(32)
        right_widget.setMaximumHeight(32)
        
        # Add badges to menu bar using QWidgetAction
        widget_action = QWidgetAction(self)
        widget_action.setDefaultWidget(right_widget)
        menubar.addSeparator()
        menubar.addAction(widget_action)
        
        # Create user icon as overlay widget (separate from menu bar for better visibility)
        self._setup_user_icon_overlay()
    
    def _setup_user_icon_overlay(self):
        """Set up user icon as an overlay widget in the top-right corner."""
        # Create professional user icon button with dropdown menu
        # Using a styled circular button with user silhouette
        self.user_button = QPushButton(self)
        self.user_button.setFixedSize(36, 36)
        self.user_button.setToolTip("User menu - Hover or click for options")
        # Create professional user icon using Unicode and styling
        # Create professional user icon - circular avatar style with gradient
        self.user_button.setText("")  # No text, we'll add icon via label
        self.user_button.setStyleSheet("""
            QPushButton {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                    stop:0 #5a5a5a, stop:0.5 #4a4a4a, stop:1 #3a3a3a);
                border: 2px solid #4d4d4d;
                border-radius: 18px;
                padding: 0px;
            }
            QPushButton:hover {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                    stop:0 #6a6a6a, stop:0.5 #5a5a5a, stop:1 #4a4a4a);
                border: 2px solid #60b8ff;
            }
            QPushButton:pressed {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.9,
                    stop:0 #3a3a3a, stop:0.5 #2a2a2a, stop:1 #1a1a1a);
                border: 2px solid #0078d4;
            }
        """)
        
        # Add professional user icon using styled text approach
        # Create a professional circular avatar with user initial
        user_icon_label = QLabel(self.user_button)
        # Use "U" for User - styled professionally as an avatar
        user_icon_label.setText("U")
        user_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        user_icon_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #ffffff;
                font-size: 18pt;
                font-weight: 700;
                border: none;
                padding: 0px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        user_icon_label.setGeometry(0, 0, 36, 36)
        user_icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        # Create user menu with exit option
        self.user_menu = QMenu(self)
        self.user_menu.setStyleSheet("""
            QMenu {
                background-color: #252526;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        exit_action = QAction("🚪 Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.user_menu.addAction(exit_action)
        
        # Timer for hover delay
        self.user_hover_timer = QTimer()
        self.user_hover_timer.setSingleShot(True)
        self.user_hover_timer.timeout.connect(self._show_user_menu)
        
        # Install event filter for hover detection
        self.user_button.installEventFilter(self)
        self.user_button.clicked.connect(self._show_user_menu)
        
        # Position the button in top-right corner
        self._position_user_icon()
        
        # Update position on window resize - install event filter on main window
        self.installEventFilter(self)
    
    def _position_user_icon(self):
        """Position user icon in the top-right corner - responsive to window size changes."""
        if not hasattr(self, 'user_button') or not self.user_button:
            return
            
        button_size = 36
        margin = 10
        
        # Get current window geometry
        window_width = self.width()
        
        # Calculate position - always top-right corner, regardless of window size
        x = window_width - button_size - margin
        y = margin  # Small margin from top
        
        # Ensure button stays within window bounds
        if x < 0:
            x = margin
        if y < 0:
            y = margin
        
        # Move and show the button
        self.user_button.move(int(x), int(y))
        self.user_button.raise_()  # Bring to front
        self.user_button.show()
    
    def _show_user_menu(self):
        """Show user menu at the user button position."""
        # Position menu below the button
        button_pos = self.user_button.mapToGlobal(self.user_button.rect().bottomLeft())
        self.user_menu.exec(button_pos)
    
    def _apply_theme(self):
        """Apply dark theme."""
        self.setStyleSheet(get_theme("dark"))
    
    def _on_file_selected(self, file_path: Path):
        """
        Handle file selection event.
        Resets all fields and updates status bar. Button will be enabled after validation.
        """
        # Reset all fields when a new file is selected
        self._reset_all_fields()
        
        self.current_file = file_path
        # Don't enable button yet - wait for validation
        self.summarize_button.setEnabled(False)
        self._update_status_bar(f"Validating file: {file_path.name}...")
    
    def _on_validation_complete(self, is_valid: bool):
        """
        Handle validation completion event.
        Enables or disables the summarize button based on validation results.
        
        Args:
            is_valid: True if file passed all validations, False otherwise
        """
        if is_valid:
            self.summarize_button.setEnabled(True)
            if self.current_file:
                self._update_status_bar(f"✓ File validated: {self.current_file.name} - Ready to process")
        else:
            self.summarize_button.setEnabled(False)
            if self.current_file:
                self._update_status_bar(f"✗ Validation failed for: {self.current_file.name} - Please check errors")
    
    def _on_file_cleared(self):
        """
        Handle file cleared event.
        Resets all fields, disables the summarize button, and clears any existing summary.
        """
        # Reset all fields when file is cleared
        self._reset_all_fields()
        
        self.current_file = None
        self.summarize_button.setEnabled(False)
        self._update_status_bar("No file selected")
    
    def _on_open_file(self):
        """Handle open file menu action."""
        self.file_upload._browse_files()
    
    def _on_summarize(self):
        """
        Handle summarize button click.
        Validates inputs and starts the summarization process in a worker thread.
        """
        # Validate that a file is selected
        if not self.current_file:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please select a PDF file first.\n\nUse 'Browse Files' or drag and drop a PDF."
            )
            return
        
        # Check if file still exists (might have been deleted)
        try:
            if not self.current_file.exists():
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"The selected file no longer exists:\n{self.current_file}"
                )
                self.file_upload.clear_selection()
                self.current_file = None
                self.summarize_button.setEnabled(False)
                return
        except Exception as e:
            logger.error(f"Error checking file existence: {str(e)}")
            QMessageBox.warning(
                self,
                "File Error",
                f"Unable to access the selected file:\n{str(e)}"
            )
            return
        
        # Disable all interactive components to prevent changes during processing
        self._disable_all_components()
        
        # Clear previous summary
        self.summary_display.clear_summary()
        
        # Get selected mode and prepare for processing
        try:
            mode = self.mode_selector.get_mode()
        except Exception as e:
            logger.error(f"Error getting mode: {str(e)}")
            QMessageBox.warning(self, "Error", "Unable to determine summarization mode.")
            # Re-enable all components on error
            self._enable_all_components()
            return
        
        # Show loading spinner
        self.loading_spinner.start()
        
        # Show progress with appropriate message (this also starts the timer)
        mode_str = "extractive" if mode == SummarizationMode.EXTRACTIVE else "abstractive"
        self.progress_indicator.show_progress(f"Starting {mode_str} summarization...")
        
        # Ensure progress indicator is visible and start timer for periodic updates
        self.progress_indicator.show()
        self.update_timer.start()
        
        # Create and start worker thread for background processing
        # This prevents the UI from freezing during long operations
        try:
            self.worker = SummarizationWorker(
                self.current_file,
                mode,
                self.extractive_summarizer,
                self.abstractive_summarizer if mode == SummarizationMode.ABSTRACTIVE else None,
                self.mode_selector  # Pass mode selector to access length preset
            )
            self.worker.progress.connect(self._on_progress_update)
            self.worker.finished.connect(self._on_summarization_finished)
            self.worker.error.connect(self._on_summarization_error)
            self.worker.start()
        except Exception as e:
            logger.error(f"Error starting worker thread: {str(e)}", exc_info=True)
            self.loading_spinner.stop()
            self.progress_indicator.hide_progress()
            # Re-enable all components on error
            self._enable_all_components()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start summarization:\n{str(e)}"
            )
    
    def _on_progress_update(self, value: int, message: str):
        """Handle progress updates."""
        self.progress_indicator.update_progress(value, message)
        # Update status bar with message and time/memory info
        self._update_status_bar(message)
    
    def _update_progress_info(self):
        """Periodically update time and memory displays during and after processing."""
        # Update time and memory if progress indicator has been initialized (has start_time)
        # This works even if the widget is temporarily hidden
        if hasattr(self.progress_indicator, 'start_time') and self.progress_indicator.start_time is not None:
            # Update time only if not in completed state (time should stop after completion)
            if not hasattr(self.progress_indicator, 'final_time') or self.progress_indicator.final_time is None:
                self.progress_indicator._update_time()
            # Always update memory to show current usage
            self.progress_indicator._update_memory()
    
    def _update_status_bar(self, message: str):
        """Update status bar with message, using semantic colors for different states."""
        # Determine message type and apply appropriate styling
        if "error" in message.lower() or "failed" in message.lower():
            # Error message - red
            self.statusBar().setStyleSheet("QStatusBar { background-color: #dc3545; color: white; }")
        elif "complete" in message.lower() or "success" in message.lower():
            # Success message - green
            self.statusBar().setStyleSheet("QStatusBar { background-color: #28a745; color: white; }")
        else:
            # Normal/processing message - blue (default)
            self.statusBar().setStyleSheet("QStatusBar { background-color: #0078d4; color: white; }")
        self.statusBar().showMessage(message)
    
    def _on_summarization_finished(self, summary: str):
        """Handle summarization completion."""
        # Stop loading spinner
        self.loading_spinner.stop()
        
        # Show completed state with final time (keeps time displayed)
        if self.progress_indicator.start_time:
            final_time = time.time() - self.progress_indicator.start_time
            self.progress_indicator.show_completed(final_time)
        
        # Keep timer running to update memory display
        # Timer will be stopped when file is selected/cleared or new summarization starts
        
        # Show summary display and set summary
        self.summary_display.show()
        self.summary_display.set_summary(summary)
        
        # Re-enable all components
        self._enable_all_components()
        
        self._update_status_bar("✓ Summary generated successfully")
        
        # Clean up worker
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None
    
    def _on_summarization_error(self, error_message: str):
        """Handle summarization error."""
        # Stop loading spinner
        self.loading_spinner.stop()
        
        # Stop the update timer
        self.update_timer.stop()
        
        self.progress_indicator.hide_progress()
        
        # Re-enable all components on error
        self._enable_all_components()
        
        self._update_status_bar("✗ Error occurred")
        
        QMessageBox.critical(self, "Error", error_message)
        
        # Clean up worker
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None
    
    def _disable_all_components(self):
        """Disable all interactive UI components during processing."""
        self.file_upload.set_enabled(False)
        self.mode_selector.set_enabled(False)
        self.summarize_button.setEnabled(False)
    
    def _enable_all_components(self):
        """Re-enable all interactive UI components after processing."""
        self.file_upload.set_enabled(True)
        self.mode_selector.set_enabled(True)
        # Only enable summarize button if a file is selected
        if self.current_file:
            self.summarize_button.setEnabled(True)
        else:
            self.summarize_button.setEnabled(False)
    
    def _reset_all_fields(self):
        """
        Reset all fields when a new file is selected or current file is removed.
        Clears summary, hides progress, resets status bar.
        """
        # Clear and hide summary display
        self.summary_display.clear_summary()
        self.summary_display.hide()
        
        # Hide progress indicator and reset timing
        self.progress_indicator.hide_progress()
        
        # Stop update timer if running
        if self.update_timer.isActive():
            self.update_timer.stop()
        
        # Reset status bar to default
        self.statusBar().setStyleSheet("QStatusBar { background-color: #0078d4; color: white; }")
        
        # Re-enable all components (they may have been disabled during processing)
        self._enable_all_components()
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
    
    def eventFilter(self, obj, event):
        """Event filter for user button hover detection."""
        if obj == self.user_button:
            if event.type() == event.Type.Enter:
                # Start timer to show menu after short delay
                self.user_hover_timer.start(300)  # 300ms delay
            elif event.type() == event.Type.Leave:
                # Cancel timer if mouse leaves
                self.user_hover_timer.stop()
        return super().eventFilter(obj, event)
    
    def resizeEvent(self, event):
        """Handle window resize to reposition user icon dynamically."""
        super().resizeEvent(event)
        if hasattr(self, 'user_button') and self.user_button:
            # Use QTimer to delay repositioning slightly to ensure accurate window size
            QTimer.singleShot(10, self._position_user_icon)
    
    def showEvent(self, event):
        """Handle window show event to position user icon when window is displayed."""
        super().showEvent(event)
        if hasattr(self, 'user_button') and self.user_button:
            # Position user icon after window is shown
            QTimer.singleShot(50, self._position_user_icon)
    
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
    
    def resizeEvent(self, event):
        """Handle window resize to reposition user icon dynamically."""
        super().resizeEvent(event)
        if hasattr(self, 'user_button') and self.user_button:
            # Use QTimer to delay repositioning slightly to ensure accurate window size
            QTimer.singleShot(10, self._position_user_icon)
    
    def showEvent(self, event):
        """Handle window show event to position user icon when window is displayed."""
        super().showEvent(event)
        if hasattr(self, 'user_button') and self.user_button:
            # Position user icon after window is shown
            QTimer.singleShot(50, self._position_user_icon)

