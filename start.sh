#!/bin/bash
# Breakout Training Studio — Single Command Startup (Linux / macOS)
# Run from the project2-breakout-studio/ directory: bash start.sh

set -e

echo ""
echo "[Breakout Studio] Installing backend dependencies..."
cd backend
pip install -r requirements.txt --quiet
echo "[Breakout Studio] Starting backend on http://localhost:8000..."
python main.py &
BACKEND_PID=$!
cd ..

sleep 2

echo "[Breakout Studio] Installing frontend dependencies..."
cd frontend
npm install --silent
echo "[Breakout Studio] Starting frontend on http://localhost:5173..."
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "[Breakout Studio] Ready!"
echo "  Open: http://localhost:5173"
echo "  Press Ctrl+C to stop."
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
