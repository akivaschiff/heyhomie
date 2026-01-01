#!/bin/bash
# Run Homie in text mode for local testing

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please create it first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Set text mode
export INTERACTION_MODE=text

# Run Homie
echo "Starting Homie in text mode..."
echo ""
python main.py
