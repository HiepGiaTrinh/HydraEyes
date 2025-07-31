#!/usr/bin/env python3
"""
WSGI entry point for H2 Factory Camera Monitoring System
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from app import app

if __name__ == "__main__":
    app.run()
