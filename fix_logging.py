# fix_logging.py - Run this once to fix logging
import sys
import io

# Set console to UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
print("✅ Console encoding set to UTF-8")
print("Restart your server with: python run.py")