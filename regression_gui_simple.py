---
name: Build Simple Windows Executable
on:
  workflow_dispatch: null
jobs:
  build-simple:
    runs-on: windows-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"
      - name: Install minimal dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy openpyxl pyinstaller
      - name: Verify basic imports
        run: |
          python -c "
          import pandas as pd
          import numpy as np
          import tkinter as tk
          print('All basic imports successful!')
          print('pandas version:', pd.__version__)
          print('numpy version:', np.__version__)
          "
      - name: Build simple executable
        run: |
          pyinstaller --onefile --name="RegressionTool" regression_gui_simple.py
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: Simple-Regression-Tool
          path: dist/RegressionTool.exe
