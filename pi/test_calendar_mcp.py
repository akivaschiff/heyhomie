#!/usr/bin/env python3
"""
Simple test to verify calendar MCP is working.
Usage: DEFAULT_CALENDAR_ID=your_email@gmail.com python3 test_calendar_mcp.py
"""

import os
import sys
from pathlib import Path

# Configuration
DEFAULT_CALENDAR_ID = os.environ.get("DEFAULT_CALENDAR_ID", "akivaschiff@gmail.com")
GOOGLE_SERVICE_ACCOUNT_PATH = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_PATH",
    str(Path(__file__).parent.parent / "secrets" / "google-calendar.json")
)

# Import after setting env
sys.path.insert(0, str(Path(__file__).parent))
from main import MCPClient

def main():
    print("Testing Calendar MCP...")
    print(f"Calendar ID: {DEFAULT_CALENDAR_ID}")
    print()

    # Initialize MCP
    mcp_path = Path(__file__).parent.parent / "mcps" / "calendar"
    server_command = ["node", str(mcp_path / "build" / "index.js")]
    env = {
        "GOOGLE_SERVICE_ACCOUNT_PATH": GOOGLE_SERVICE_ACCOUNT_PATH,
        "DEFAULT_CALENDAR_ID": DEFAULT_CALENDAR_ID
    }

    try:
        client = MCPClient(server_command, env)
        client.start()

        print(f"✅ MCP started with {len(client.tools)} tools")

        # Test list_calendars
        result = client.call_tool("list_calendars", {})
        print(f"\n📅 Calendars: {result}")

        # Test list_events
        result = client.call_tool("list_events", {"maxResults": 5})
        print(f"\n📆 Upcoming events: {result}")

        client.stop()
        print("\n✅ Test complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
