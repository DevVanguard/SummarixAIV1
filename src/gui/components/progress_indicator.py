"""Progress indicator component for showing processing status."""

import logging
import os
import time
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class ProgressIndicatorWidget(QWidget):
    """Widget for displaying progress and status."""
    
    def __init__(self, parent=None):
        """Initialize progress indicator widget."""
        super().__init__(parent)
        self.start_time: Optional[float] = None
        self.final_time: Optional[float] = None
        self._setup_ui()
        self.hide()
    
    def _setup_ui(self):
        """
        Set up the UI components following HCI principles.
        Compact design with efficient space usage.
        """
        layout = QVBoxLayout()
        layout.setSpacing(6)  # Tighter spacing
        layout.setContentsMargins(0, 4, 0, 4)  # Minimal margins
        
        # Status label - compact
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #e0e0e0; font-size: 9pt; padding: 2px 0px;")
        
        # Progress bar - compact height
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(20)  # Compact height
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                font-size: 8pt;
            }
        """)
        
        # Info bar with time and memory - horizontal layout
        info_layout = QHBoxLayout()
        info_layout.setSpacing(12)
        info_layout.setContentsMargins(0, 2, 0, 0)
        
        # Processing time label
        self.time_label = QLabel("⏱ Time: --")
        self.time_label.setStyleSheet("color: #808080; font-size: 8pt;")
        
        # Memory usage label
        self.memory_label = QLabel("💾 Memory: --")
        self.memory_label.setStyleSheet("color: #808080; font-size: 8pt;")
        
        info_layout.addWidget(self.time_label)
        info_layout.addStretch()
        info_layout.addWidget(self.memory_label)
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(info_layout)
        
        self.setLayout(layout)
    
    def show_progress(self, message: str = "Processing..."):
        """Show progress indicator and start timing."""
        self.status_label.setText(message)
        self.progress_bar.setValue(0)
        self.start_time = time.time()
        self.final_time = None  # Reset final time when starting new progress
        self._update_time()
        self._update_memory()
        self.show()
    
    def update_progress(self, value: int, message: Optional[str] = None):
        """
        Update progress value and refresh time/memory displays.
        
        Args:
            value: Progress value (0-100)
            message: Optional status message
        """
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
        self._update_time()
        self._update_memory()
    
    def set_indeterminate(self, message: str = "Processing..."):
        """Set progress bar to indeterminate mode."""
        self.status_label.setText(message)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
    
    def set_determinate(self, max_value: int = 100):
        """Set progress bar to determinate mode."""
        self.progress_bar.setRange(0, max_value)
        self.progress_bar.setValue(0)
    
    def show_completed(self, final_time: Optional[float] = None):
        """
        Show completed state with final time displayed.
        Hides progress bar but keeps time and memory visible.
        Memory will continue to update periodically.
        
        Args:
            final_time: Final elapsed time in seconds (if None, uses current elapsed time)
        """
        if self.start_time is not None:
            if final_time is None:
                final_time = time.time() - self.start_time
            # Store final time for display
            self.final_time = final_time
            self.time_label.setText(f"⏱ Time: {final_time:.1f}s")
            self._update_memory()
            # Hide progress bar and status label, but keep time/memory visible
            self.progress_bar.hide()
            self.status_label.hide()
            self.show()
    
    def _update_time(self):
        """Update the processing time display."""
        # If we're in completed state, just show the final time (don't update)
        if hasattr(self, 'final_time') and self.final_time is not None:
            # Time has stopped - just display the final time
            self.time_label.setText(f"⏱ Time: {self.final_time:.1f}s")
            return
        
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed:.1f}s"
        
        # Estimate remaining time based on progress (if progress bar is visible and has value)
        if self.progress_bar.isVisible() and self.progress_bar.maximum() > 0:
            progress = self.progress_bar.value()
            if progress > 0 and progress < 100:
                estimated_total = elapsed / (progress / 100.0)
                remaining = estimated_total - elapsed
                estimated_str = f" | Est: {remaining:.1f}s"
            else:
                estimated_str = ""
        else:
            # No progress info available, just show elapsed time
            estimated_str = ""
        
        self.time_label.setText(f"⏱ Time: {elapsed_str}{estimated_str}")
    
    def hide_progress(self):
        """Hide progress indicator and reset timing."""
        self.hide()
        self.progress_bar.setValue(0)
        self.progress_bar.show()  # Show progress bar again for next use
        self.status_label.show()  # Show status label again for next use
        self.status_label.setText("")
        self.start_time = None
        self.final_time = None  # Reset final time
        self.time_label.setText("⏱ Time: --")
        self.memory_label.setText("💾 Memory: --")
    
    def _update_memory(self):
        """Update the memory usage display using built-in os module."""
        try:
            # Try to get memory info using psutil if available (it's in requirements)
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_label.setText(f"💾 Memory: {memory_mb:.1f} MB")
            except ImportError:
                # Fallback: Use os module for basic info (Windows-specific)
                if os.name == 'nt':
                    try:
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        process_handle = kernel32.GetCurrentProcess()
                        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                            _fields_ = [
                                ("cb", ctypes.c_ulong),
                                ("PageFaultCount", ctypes.c_ulong),
                                ("PeakWorkingSetSize", ctypes.c_size_t),
                                ("WorkingSetSize", ctypes.c_size_t),
                                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                                ("PagefileUsage", ctypes.c_size_t),
                                ("PeakPagefileUsage", ctypes.c_size_t),
                            ]
                        pmc = PROCESS_MEMORY_COUNTERS()
                        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                        if kernel32.GetProcessMemoryInfo(process_handle, ctypes.byref(pmc), pmc.cb):
                            memory_mb = pmc.WorkingSetSize / (1024 * 1024)
                            self.memory_label.setText(f"💾 Memory: {memory_mb:.1f} MB")
                        else:
                            self.memory_label.setText("💾 Memory: N/A")
                    except Exception:
                        self.memory_label.setText("💾 Memory: N/A")
                else:
                    # For Unix-like systems, use resource module
                    try:
                        import resource
                        memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
                        if memory_kb > 1000000:  # Likely bytes (macOS)
                            memory_mb = memory_kb / (1024 * 1024)
                        else:  # Likely KB (Linux)
                            memory_mb = memory_kb / 1024
                        self.memory_label.setText(f"💾 Memory: {memory_mb:.1f} MB")
                    except Exception:
                        self.memory_label.setText("💾 Memory: N/A")
        except Exception as e:
            logger.debug(f"Could not get memory info: {str(e)}")
            self.memory_label.setText("💾 Memory: N/A")

