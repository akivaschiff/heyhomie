#!/usr/bin/env python3
"""
Simple test to verify system MCP is working.
Usage: python3 test_system_mcp.py
"""

import sys
from pathlib import Path

# Import after setting env
sys.path.insert(0, str(Path(__file__).parent))
from main import MCPClient

def main():
    print("Testing System MCP...")
    print()

    # Initialize MCP
    mcp_path = Path(__file__).parent.parent / "mcps" / "system"
    server_command = ["node", str(mcp_path / "build" / "index.js")]
    env = {}

    try:
        client = MCPClient(server_command, env)
        client.start()

        print(f"✅ MCP started with {len(client.tools)} tools")
        print(f"   Tools: {[t['name'] for t in client.tools]}")

        # Test get_datetime
        print("\n🕐 Testing get_datetime...")
        result = client.call_tool("get_datetime", {})
        print(f"Result: {result}")

        # Test get_volume (may fail on non-Linux systems)
        print("\n🔊 Testing get_volume...")
        result = client.call_tool("get_volume", {})
        print(f"Result: {result}")

        # Test set_volume (may fail on non-Linux systems)
        # Commenting out by default to avoid changing volume during tests
        # print("\n🔊 Testing set_volume...")
        # result = client.call_tool("set_volume", {"volume": 50})
        # print(f"Result: {result}")

        client.stop()
        print("\n✅ Test complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
