#!/bin/bash

echo "================================================================================"
echo "🤖 AGENTIC UNDERWRITING ASSISTANT - DEMO"
echo "================================================================================"
echo

echo "🔍 Starting services..."
docker-compose up -d

echo
echo "⏳ Waiting for services to start (30 seconds)..."
sleep 30

echo
echo "🚀 Running demo..."
python3 demo.py

echo
echo "🛑 Stopping services..."
docker-compose down

echo
echo "✅ Demo completed!"
