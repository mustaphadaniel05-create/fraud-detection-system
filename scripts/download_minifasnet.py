#!/usr/bin/env python3
"""
Download MiniFASNet pretrained weights for anti-spoofing.
Required for pretrained_antispoof_service.py

Run: python scripts/download_minifasnet.py
"""

import os
import sys
import urllib.request
import urllib.error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create models directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "2.7_80x80_MiniFASNetV2.pth")

# Multiple mirror URLs
URLS = [
    "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
    "https://huggingface.co/spaces/akhiljalagam/Silent-Face-Anti-Spoofing/resolve/main/2.7_80x80_MiniFASNetV2.pth",
]


def download_file(url, destination):
    """Download file with progress bar."""
    print(f"Downloading from: {url}")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))
            sys.stdout.write(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def main():
    print("=" * 60)
    print("MiniFASNet Anti-Spoof Model Downloader")
    print("=" * 60)
    print(f"Model will be saved to: {MODEL_PATH}")
    print()
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model already exists!")
        print(f"   Location: {MODEL_PATH}")
        print(f"   Size: {size_mb:.1f} MB")
        return True
    
    # Try each URL
    for url in URLS:
        if download_file(url, MODEL_PATH):
            if os.path.exists(MODEL_PATH):
                size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
                print(f"✅ Model saved! Size: {size_mb:.1f} MB")
                return True
        print("Trying next mirror...")
    
    print("\n❌ Failed to download model from all sources.")
    print("\n📌 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("   1. Go to: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing")
    print("   2. Download: 2.7_80x80_MiniFASNetV2.pth")
    print(f"   3. Save to: {MODEL_PATH}")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)