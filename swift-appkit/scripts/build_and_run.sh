#!/bin/bash
set -e

# このスクリプトは swift-appkit ディレクトリから実行されることを想定
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

APP_NAME="EHR-Agent.app"
APP_BUNDLE_ID="com.ehr-agentic-toolkit.EHR-Agent"
APP_SRC="./$APP_NAME"
APP_DST="$HOME/Applications/$APP_NAME"

echo "Building EHR-Agent..."
swiftc -target arm64-apple-macos14.0 -o /tmp/EHR-Agent main.swift

echo "Copying binary to local app bundle..."
cp /tmp/EHR-Agent "$APP_SRC/Contents/MacOS/EHR-Agent"

echo "Copying template images to bundle Resources..."
mkdir -p "$APP_SRC/Contents/Resources/match_templates"
cp match_templates/*.png "$APP_SRC/Contents/Resources/match_templates/"

echo "Stripping extended attributes..."
xattr -cr "$APP_SRC"

echo "Re-signing app..."
codesign --force --deep --sign - "$APP_SRC"

echo "Installing to ~/Applications/..."
rm -rf "$APP_DST"
cp -R "$APP_SRC" "$APP_DST"

echo "Registering with Launch Services..."
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_DST"

echo "Resetting permissions..."
./scripts/reset_permissions.sh

echo "Opening app and System Settings..."
open "$APP_DST"
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

echo "Build complete. Please grant permissions in System Settings."
