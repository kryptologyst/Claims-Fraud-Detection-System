"""Setup script for the fraud detection system."""

#!/usr/bin/env python3
"""Setup script for Claims Fraud Detection System."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🛡️ Claims Fraud Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "models", "logs", "assets", "tests"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install pre-commit hooks (optional)
    if Path(".pre-commit-config.yaml").exists():
        if run_command("pre-commit install", "Installing pre-commit hooks"):
            print("✅ Pre-commit hooks installed")
        else:
            print("⚠️ Pre-commit hooks installation failed (optional)")
    
    # Run basic tests
    if run_command("python -m pytest tests/ -v", "Running basic tests"):
        print("✅ All tests passed")
    else:
        print("⚠️ Some tests failed (check dependencies)")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train the models: python scripts/train.py")
    print("2. Launch the demo: streamlit run demo/app.py")
    print("3. Read the documentation: README.md")
    print("\n⚠️ Remember: This system is for educational purposes only!")
    print("=" * 50)


if __name__ == "__main__":
    main()
