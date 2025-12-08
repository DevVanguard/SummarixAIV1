"""Validation utilities for file and system checks."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.utils.config import Config

logger = logging.getLogger(__name__)


def validate_file_size(file_path: Path) -> Tuple[bool, Optional[str], str]:
    """
    Validate file size against configured limits.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, warning_message, error_message)
        - is_valid: True if file size is acceptable
        - warning_message: Warning message if file is large (or None)
        - error_message: Error message if file is too large (or None)
    """
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Check hard limit
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            error_msg = (
                f"File too large ({file_size_mb:.2f} MB). "
                f"Maximum allowed: {Config.MAX_FILE_SIZE_MB} MB. "
                f"Please use a smaller file or split the PDF."
            )
            return False, None, error_msg
        
        # Check warning threshold
        if file_size_mb > Config.WARNING_FILE_SIZE_MB:
            warning_msg = (
                f"Large file ({file_size_mb:.2f} MB). "
                f"Processing may take longer than usual."
            )
            return True, warning_msg, None
        
        # File size is OK
        size_str = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_size_mb * 1024:.1f} KB"
        return True, None, None
        
    except Exception as e:
        logger.error(f"Error validating file size: {str(e)}")
        return False, None, f"Error checking file size: {str(e)}"


def check_available_memory() -> Tuple[bool, Optional[str], float]:
    """
    Check available system memory.
    
    Returns:
        Tuple of (has_sufficient_memory, warning_message, available_mb)
        - has_sufficient_memory: True if enough memory available
        - warning_message: Warning if memory is low (or None)
        - available_mb: Available memory in MB
    """
    try:
        # Try to use psutil if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            if available_mb < Config.MIN_AVAILABLE_MEMORY_MB:
                warning_msg = (
                    f"Low available memory ({available_mb:.0f} MB). "
                    f"Processing large files may fail."
                )
                return False, warning_msg, available_mb
            
            return True, None, available_mb
            
        except ImportError:
            # Fallback for Windows without psutil
            if os.name == 'nt':
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    
                    mem_status = MEMORYSTATUSEX()
                    mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
                    
                    available_mb = mem_status.ullAvailPhys / (1024 * 1024)
                    
                    if available_mb < Config.MIN_AVAILABLE_MEMORY_MB:
                        warning_msg = (
                            f"Low available memory ({available_mb:.0f} MB). "
                            f"Processing large files may fail."
                        )
                        return False, warning_msg, available_mb
                    
                    return True, None, available_mb
                    
                except Exception as e:
                    logger.warning(f"Could not check memory: {str(e)}")
                    # Assume OK if we can't check
                    return True, "Could not check memory availability", 0.0
            else:
                # Unix-like systems - use resource module
                try:
                    import resource
                    # This gives max RSS, not available memory
                    # For now, assume OK if we can't check properly
                    return True, "Memory check not available on this system", 0.0
                except Exception:
                    return True, "Memory check not available", 0.0
                    
    except Exception as e:
        logger.error(f"Error checking memory: {str(e)}")
        return True, "Could not check memory", 0.0


def estimate_processing_memory(file_size_mb: float) -> float:
    """
    Estimate memory needed for processing a file.
    
    Args:
        file_size_mb: File size in MB
        
    Returns:
        Estimated memory needed in MB
    """
    return file_size_mb * Config.MEMORY_MULTIPLIER


def validate_memory_for_file(file_path: Path) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate that there's enough memory to process the file.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Tuple of (has_sufficient_memory, warning_message, error_message)
    """
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        estimated_memory = estimate_processing_memory(file_size_mb)
        
        has_memory, memory_warning, available_mb = check_available_memory()
        
        if not has_memory:
            error_msg = (
                f"Insufficient memory. "
                f"Estimated need: {estimated_memory:.0f} MB, "
                f"Available: {available_mb:.0f} MB. "
                f"Please close other applications or use a smaller file."
            )
            return False, None, error_msg
        
        if available_mb > 0 and estimated_memory > available_mb * 0.8:
            warning_msg = (
                f"Processing may use significant memory ({estimated_memory:.0f} MB estimated). "
                f"Available: {available_mb:.0f} MB."
            )
            return True, warning_msg, None
        
        return True, None, None
        
    except Exception as e:
        logger.error(f"Error validating memory for file: {str(e)}")
        return True, "Could not estimate memory requirements", None

