#!/bin/bash
set -e

BUNDLE_ID="com.ehr-agentic-toolkit.EHR-Agent"

echo "Resetting TCC permissions for EHR-Agent (Bundle ID: $BUNDLE_ID)..."

# Accessibility 権限をリセット
sudo tccutil reset Accessibility "$BUNDLE_ID" || true

# Screen Capture 権限をリセット
sudo tccutil reset ScreenCapture "$BUNDLE_ID" || true

echo "Done. Please restart EHR-Agent and grant permissions again."
