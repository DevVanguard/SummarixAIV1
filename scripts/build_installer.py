"""Script to build installer using PyInstaller."""

import logging
import os
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


def create_spec_file():
    """Create PyInstaller spec file."""
    from pathlib import Path as PathLib
    
    # Check if icon exists
    icon_path = PathLib(__file__).parent.parent / "resources" / "icons" / "app.ico"
    icon_str = f"'{icon_path}'" if icon_path.exists() else "None"
    
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('resources/models', 'resources/models'),
        ('resources/icons', 'resources/icons'),
    ],
    hiddenimports=[
        'transformers',
        'torch',
        'networkx',
        'nltk',
        'sklearn',
        'fitz',
        'PyQt6',
        'docx',
        'reportlab',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SummarixAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=""" + icon_str + """,
)
"""
    
    spec_path = Path(__file__).parent.parent / "SummarixAI.spec"
    spec_path.write_text(spec_content)
    logger.info(f"Created spec file: {spec_path}")
    return spec_path


def build_executable():
    """Build executable using PyInstaller."""
    try:
        import PyInstaller.__main__
    except ImportError:
        logger.error("PyInstaller not installed. Install with: pip install pyinstaller")
        return False
    
    # Create spec file
    spec_path = create_spec_file()
    
    # Build
    logger.info("Building executable with PyInstaller...")
    logger.info("This may take several minutes...")
    
    try:
        PyInstaller.__main__.run([
            str(spec_path),
            '--clean',
            '--noconfirm',
        ])
        
        logger.info("Build completed successfully!")
        logger.info(f"Executable location: dist/SummarixAI")
        return True
        
    except Exception as e:
        logger.error(f"Build failed: {str(e)}")
        return False


def create_installer_windows():
    """Create Windows installer using Inno Setup (if available)."""
    try:
        import subprocess
        
        # Check if Inno Setup is available
        inno_paths = [
            r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            r"C:\Program Files\Inno Setup 6\ISCC.exe",
        ]
        
        inno_script = Path(__file__).parent.parent / "installer.iss"
        
        # Create Inno Setup script
        iss_content = """[Setup]
AppName=SummarixAI
AppVersion=1.0.0
DefaultDirName={pf}\\SummarixAI
DefaultGroupName=SummarixAI
OutputDir=installer
OutputBaseFilename=SummarixAI-Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin

[Files]
Source: "dist\\SummarixAI\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\\SummarixAI"; Filename: "{app}\\SummarixAI.exe"
Name: "{commondesktop}\\SummarixAI"; Filename: "{app}\\SummarixAI.exe"

[Run]
Filename: "{app}\\SummarixAI.exe"; Description: "Launch SummarixAI"; Flags: nowait postinstall skipifsilent
"""
        
        if not inno_script.exists():
            inno_script.write_text(iss_content)
            logger.info(f"Created Inno Setup script: {inno_script}")
        
        # Try to run Inno Setup compiler
        for inno_path in inno_paths:
            if Path(inno_path).exists():
                logger.info("Building Windows installer with Inno Setup...")
                subprocess.run([inno_path, str(inno_script)])
                logger.info("Installer created successfully!")
                return True
        
        logger.warning("Inno Setup not found. Skipping installer creation.")
        logger.info("You can manually create an installer using the installer.iss file")
        return False
        
    except Exception as e:
        logger.warning(f"Could not create installer: {str(e)}")
        return False


def main():
    """Main build function."""
    logger.info("Starting build process...")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Check if models exist
    if not Config.MODEL_CACHE_DIR.exists():
        logger.warning(
            f"Models not found at {Config.MODEL_CACHE_DIR}. "
            "Please run scripts/download_models.py first."
        )
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Build executable
    if not build_executable():
        logger.error("Build failed")
        sys.exit(1)
    
    # Create installer (Windows only)
    if sys.platform == "win32":
        create_installer_windows()
    
    logger.info("Build process completed!")


if __name__ == "__main__":
    main()

