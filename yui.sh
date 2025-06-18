#!/bin/bash

# Yui Voice Agent - Clean Output Version
# This script runs the voice agent while filtering out noisy audio warnings

echo "🎤 Starting Yui Voice Agent (Clean Output Mode)..."
echo ""

# Run the voice agent and filter out specific audio warnings
bun run index.ts "$@" 2>&1 | grep -v -E "(buffer underflow|mpg123.*warning|Didn't have any audio data)" 