#!/usr/bin/env python3
"""
Main entry point for the fraud detection API.
FIXED: Unicode/emoji handling for Windows console
"""

import sys
import io
import logging

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging without emojis for Windows compatibility
class SafeFormatter(logging.Formatter):
    """Formatter that removes or replaces emojis for Windows console"""
    def format(self, record):
        # Replace emojis with text equivalents
        if record.msg:
            if isinstance(record.msg, str):
                record.msg = (record.msg
                    .replace('✅', '[OK]')
                    .replace('❌', '[FAIL]')
                    .replace('🚫', '[BLOCK]')
                    .replace('⚠️', '[WARN]')
                    .replace('📱', '[REQUEST]')
                    .replace('📏', '[SIZE]')
                    .replace('🔍', '[SEARCH]')
                    .replace('👤', '[FACE]')
                    .replace('🎭', '[DEEPFAKE]')
                    .replace('📸', '[PHOTO]')
                    .replace('🛡️', '[SHIELD]')
                    .replace('🔴', '[RED]')
                    .replace('✅✅✅', '[OK]')
                    .replace('❌❌❌', '[FAIL]')
                    .replace('🚫🚫🚫', '[BLOCK]'))
        return super().format(record)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Apply safe formatter to root logger
for handler in logging.root.handlers:
    handler.setFormatter(SafeFormatter())

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Development only – do NOT use in production
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=app.config.get("DEBUG", False),
        use_reloader=False,
        threaded=True,
    )