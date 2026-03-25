import sys
import os

# Make the repo root importable so test files can do `import _geometry`, `from metrics import ...`, etc.
sys.path.insert(0, os.path.dirname(__file__))
