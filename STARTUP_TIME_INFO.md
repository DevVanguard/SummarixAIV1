# SummarixAI Startup Time Guide

## Expected Startup Times

### First Launch (Cold Start)
- **Initial extraction**: 30-60 seconds
  - PyInstaller extracts ~500MB of files to temp directory
  - Happens only on first run or after temp cleanup
  
- **Application window appears**: 10-20 seconds after extraction
  - GUI initialization
  - Loading PyQt6, PyTorch, transformers libraries
  
- **Ready to use**: **40-80 seconds total** (first time)

### Subsequent Launches (Warm Start)
- **Application window appears**: 10-30 seconds
  - Temp files already extracted
  - Faster library loading
  
- **Ready to use**: **10-30 seconds**

## What Happens During Startup

### Phase 1: File Extraction (First Run Only)
```
Extracting bundled files...
├── Python runtime (~50MB)
├── PyTorch (~400MB)
├── Transformers (~100MB)
├── PyQt6 (~50MB)
└── Models (~233MB)
```
**Time**: 30-60 seconds (only first run)

### Phase 2: Library Loading
```
Loading libraries...
├── PyQt6 GUI framework
├── PyTorch (CPU version)
├── Transformers library
└── Other dependencies
```
**Time**: 5-15 seconds

### Phase 3: Application Ready
- GUI window appears
- Ready to upload PDFs
- Extractive mode: **Immediately available**
- Abstractive mode: **Requires model loading** (see below)

## Abstractive Mode Model Loading

When user selects **Abstractive mode** for the first time:

- **Model loading**: 15-30 seconds
  - Loading T5-small model (~233MB)
  - Initializing tokenizer
  - Setting up inference pipeline
  
- **Subsequent uses**: 5-10 seconds (model cached in memory)

## Performance Tips

### To Speed Up Startup:
1. **Keep temp directory**: Don't clear Windows temp files
2. **SSD vs HDD**: 
   - SSD: 30-50 seconds (first run)
   - HDD: 60-90 seconds (first run)
3. **RAM**: More RAM = faster loading
4. **Antivirus**: May slow down first extraction

### What Affects Speed:
- **CPU**: Faster CPU = faster extraction
- **Storage**: SSD much faster than HDD
- **RAM**: 8GB+ recommended
- **Antivirus**: Real-time scanning can slow extraction

## Typical User Experience

### First Time User:
```
1. Double-click SummarixAI.exe
2. Wait 30-60 seconds (extraction happening)
3. Window appears
4. Ready to use!
```

### Regular User:
```
1. Double-click SummarixAI.exe
2. Wait 10-30 seconds
3. Window appears
4. Ready to use!
```

## Troubleshooting Slow Startup

### If startup takes >2 minutes:
- Check Windows Defender/Antivirus (may be scanning)
- Check disk space (need ~1GB free in temp)
- Check CPU usage (other programs running?)
- Try running as Administrator

### If window doesn't appear:
- Check Task Manager for `SummarixAI.exe` process
- May be extracting files (wait longer)
- Check Windows Event Viewer for errors

## Progress Indicators

Currently, the application shows:
- **No progress bar during extraction** (PyInstaller handles this)
- **Window appears when ready**

Future enhancement: Could add splash screen with progress indicator.

## Summary

| Scenario | Time |
|----------|------|
| First launch (cold start) | 40-80 seconds |
| Subsequent launches | 10-30 seconds |
| Abstractive mode (first use) | +15-30 seconds |
| Abstractive mode (cached) | +5-10 seconds |

**Bottom line**: Be patient on first run (30-60 seconds), then it's much faster!

