"""Pre-build checklist script - Run this before building the exe."""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"  Python {version.major}.{version.minor} - Need 3.8+")
        return False

def check_dependencies():
    """Check if all dependencies are installed."""
    print("\n✓ Checking dependencies...")
    # Map package names to their actual import names (case-sensitive)
    required = {
        'PyQt6': 'PyQt6',           # Case-sensitive - must be 'PyQt6'
        'fitz': 'fitz',             # PyMuPDF imports as 'fitz'
        'transformers': 'transformers',
        'torch': 'torch',
        'networkx': 'networkx',
        'nltk': 'nltk',
        'sklearn': 'sklearn',       # scikit-learn imports as 'sklearn'
        'docx': 'docx',             # python-docx imports as 'docx'
        'reportlab': 'reportlab',
        'PyInstaller': 'PyInstaller'  # Case-sensitive - must be 'PyInstaller'
    }
    missing = []
    for package_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  {package_name} - OK")
        except ImportError:
            print(f"  {package_name} - MISSING")
            missing.append(package_name)
    
    if missing:
        print(f"\n✗ Missing: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    return True

def check_nltk_data():
    """Check and download NLTK data."""
    print("\n✓ Checking NLTK data...")
    try:
        import nltk
        # Find NLTK data path
        nltk_paths = nltk.data.path
        print(f"  NLTK data paths: {nltk_paths}")
        
        # Check punkt
        punkt_ok = False
        try:
            nltk.data.find('tokenizers/punkt_tab/english.pickle')
            print("  punkt_tab - OK")
            punkt_ok = True
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt/english.pickle')
                print("  punkt - OK")
                punkt_ok = True
            except LookupError:
                print("  punkt - MISSING, downloading...")
                nltk.download('punkt_tab', quiet=True)
                print("  punkt_tab - Downloaded")
                punkt_ok = True
        
        # Check stopwords
        stopwords_ok = False
        try:
            nltk.data.find('corpora/stopwords/english.zip')
            print("  stopwords - OK")
            stopwords_ok = True
        except LookupError:
            print("  stopwords - MISSING, downloading...")
            nltk.download('stopwords', quiet=True)
            print("  stopwords - Downloaded")
            stopwords_ok = True
        
        return punkt_ok and stopwords_ok
    except Exception as e:
        print(f"  Error: {e}")
        return False

def check_models():
    """Check if models are downloaded."""
    print("\n✓ Checking AI models...")
    sys.path.insert(0, str(Path(__file__).parent))
    from src.utils.config import Config
    
    model_path = Config.MODEL_CACHE_DIR
    if model_path.exists():
        files = list(model_path.glob('*'))
        if len(files) > 0:
            print(f"  Models found at: {model_path}")
            print(f"  Files: {len(files)}")
            return True
        else:
            print(f"  Model directory empty: {model_path}")
    else:
        print(f"  Models not found at: {model_path}")
    
    print("  Run: python scripts/download_models.py")
    return False

def check_resources():
    """Check if resource directories exist."""
    print("\n✓ Checking resources...")
    base = Path(__file__).parent
    resources = base / "resources"
    models = resources / "models"
    icons = resources / "icons"
    
    if not resources.exists():
        print(f"  Creating: {resources}")
        resources.mkdir(parents=True)
    
    if not models.exists():
        print(f"  Creating: {models}")
        models.mkdir(parents=True)
    
    if not icons.exists():
        print(f"  Creating: {icons}")
        icons.mkdir(parents=True)
    
    print("  Resources - OK")
    return True

def main():
    """Run all checks."""
    print("=" * 60)
    print("PRE-BUILD CHECKLIST")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("NLTK Data", check_nltk_data),
        ("AI Models", check_models),
        ("Resources", check_resources),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_ok = False
    
    if all_ok:
        print("\n✓ All checks passed! Ready to build.")
        print("\nNext step: python scripts/build_installer.py")
    else:
        print("\n✗ Some checks failed. Fix issues above before building.")
        sys.exit(1)

if __name__ == "__main__":
    main()