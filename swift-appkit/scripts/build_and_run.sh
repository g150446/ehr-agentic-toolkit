#!/bin/bash
set -e

# このスクリプトは swift-appkit ディレクトリから実行されることを想定
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Building EHR-Agent..."
swiftc -o /tmp/EHR-Agent main.swift

echo "Copying binary to app bundle..."
cp /tmp/EHR-Agent EHR-Agent.app/Contents/MacOS/EHR-Agent

echo "Re-signing app..."
codesign --force --deep --sign - EHR-Agent.app

echo "Resetting permissions..."
./scripts/reset_permissions.sh

echo "Opening app and System Settings..."
open EHR-Agent.app
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

echo "Build complete. Please grant permissions in System Settings."
