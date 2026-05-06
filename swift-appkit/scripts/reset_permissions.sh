#!/bin/bash
set -e

BUNDLE_ID="com.ehr-agentic-toolkit.EHR-Agent"
TCC_DB="$HOME/Library/Application Support/com.apple.TCC/TCC.db"

echo "Resetting TCC permissions for EHR-Agent (Bundle ID: $BUNDLE_ID)..."

# Detect ghost entries (entries without proper bundle ID or conflicting entries)
echo "Checking for ghost entries in TCC database..."
if [ -f "$TCC_DB" ]; then
    GHOST_ENTRIES=$(sqlite3 "$TCC_DB" "SELECT client, service FROM access WHERE client LIKE '%EHR-Agent%' AND client != '$BUNDLE_ID';" 2>/dev/null || true)
    if [ -n "$GHOST_ENTRIES" ]; then
        echo "WARNING: Found ghost entries for EHR-Agent:"
        echo "$GHOST_ENTRIES"
        echo "These entries may cause permission issues. Cleaning up..."
        # Reset all EHR-Agent related entries to clean up ghosts
        sqlite3 "$TCC_DB" "DELETE FROM access WHERE client LIKE '%EHR-Agent%';" 2>/dev/null || true
        echo "Ghost entries cleaned up."
    else
        echo "No ghost entries found."
    fi
fi

# Reset permissions for the specified bundle ID
sudo tccutil reset Accessibility "$BUNDLE_ID" || true
sudo tccutil reset ScreenCapture "$BUNDLE_ID" || true

echo "Done. Please restart EHR-Agent and grant permissions again."
