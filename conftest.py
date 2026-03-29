import os
import sys

# Add the src/ directory to sys.path so `recsys` package is importable
# without requiring `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
