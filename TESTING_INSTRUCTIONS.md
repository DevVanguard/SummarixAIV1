# Testing SummarixAI on Another Computer

## Executable Location

The executable has been built and is located at:
```
dist/SummarixAI.exe
```

## Transfer Instructions

### Option 1: USB Drive (Recommended)
1. Copy the entire `dist` folder to a USB drive
2. On the other computer, copy the folder to Desktop or any location
3. Double-click `SummarixAI.exe` to run

### Option 2: Cloud Storage
1. Zip the `dist` folder
2. Upload to Google Drive, Dropbox, OneDrive, etc.
3. Download on the other computer
4. Extract and run `SummarixAI.exe`

### Option 3: Network Share
1. Share the `dist` folder on your network
2. Access from the other computer
3. Copy and run

## Important Notes

### Model Files
- The executable includes model files in `resources/models/`
- If models are missing, the app will prompt you to download them
- Models can be downloaded by running: `python scripts/download_models.py` (if Python is installed)

### First Run
- First launch may take 10-30 seconds to initialize
- Abstractive mode requires model loading (first time only)
- Extractive mode works immediately

### System Requirements
- Windows 10/11 (64-bit)
- 4GB RAM minimum (8GB recommended)
- No Python installation required
- No internet connection required (after models are included)

### Troubleshooting

**"Windows protected your PC" warning:**
- Click "More info" → "Run anyway"
- This is normal for unsigned executables

**Application won't start:**
- Check Windows Defender/Antivirus (may block new executables)
- Try running as Administrator
- Check if all DLLs are present in the dist folder

**"Model not found" error:**
- Models should be in `resources/models/t5-small/`
- If missing, download models separately or use extractive mode only

**Slow performance:**
- Abstractive mode is slower (requires model loading)
- Use extractive mode for faster results
- Close other applications to free up RAM

## Testing Checklist

- [ ] Application launches successfully
- [ ] PDF file upload works (drag-and-drop or browse)
- [ ] Extractive summarization works
- [ ] Abstractive summarization works (if models included)
- [ ] Summary display shows correctly
- [ ] Export functions work (TXT, PDF, DOCX)
- [ ] No internet connection required

## File Size

The executable is approximately **800MB-1.5GB** (includes PyTorch, transformers, and all dependencies).

## Support

If you encounter issues:
1. Check the error message
2. Verify all files are present in the dist folder
3. Try running from command line to see error messages:
   ```
   dist\SummarixAI.exe
   ```

