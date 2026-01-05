"""MCP (Model Context Protocol) client for communicating with MCP servers."""

import json
import os
import subprocess


class MCPClient:
    """Client for communicating with MCP servers via stdio."""

    def __init__(self, server_command: list[str], env: dict = None):
        """Initialize MCP client with server command.

        Args:
            server_command: Command to start the MCP server (e.g., ["node", "build/index.js"])
            env: Environment variables to pass to the server
        """
        self.server_command = server_command
        self.env = env or {}
        self.process = None
        self.tools = []
        self.message_id = 0

    def start(self):
        """Start the MCP server process."""
        env = os.environ.copy()
        env.update(self.env)

        try:
            self.process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1
            )

            # Initialize and list tools
            self._initialize()
            self._list_tools()
        except Exception:
            # Clean up if initialization fails
            if self.process:
                self.process.kill()
                self.process = None
            raise

    def _initialize(self):
        """Send initialize request to the MCP server."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "homie",
                    "version": "1.0.0"
                }
            }
        }
        self._send_request(init_request)
        response = self._read_response()
        if "error" in response:
            raise RuntimeError(f"MCP initialize failed: {response['error']}")

        # Send initialized notification
        initialized_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        self._send_request(initialized_notif)

    def _list_tools(self):
        """Fetch available tools from the MCP server."""
        list_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        self._send_request(list_request)
        response = self._read_response()

        if "error" in response:
            raise RuntimeError(f"MCP tools/list failed: {response['error']}")

        self.tools = response.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as a dictionary
        """
        call_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        self._send_request(call_request)
        response = self._read_response()

        if "error" in response:
            return {"error": response["error"]}

        return response.get("result", {})

    def get_anthropic_tools(self) -> list[dict]:
        """Convert MCP tools to Anthropic tool format."""
        anthropic_tools = []
        for tool in self.tools:
            # Convert MCP tool schema to Anthropic format
            input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": input_schema
            })
        return anthropic_tools

    def _next_id(self):
        """Generate next message ID."""
        self.message_id += 1
        return self.message_id

    def _send_request(self, request: dict):
        """Send a JSON-RPC request to the server."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server not started")

        message = json.dumps(request) + "\n"
        self.process.stdin.write(message)
        self.process.stdin.flush()

    def _read_response(self, timeout: float = 30.0) -> dict:
        """Read a JSON-RPC response from the server with timeout."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("MCP server not started")

        import select
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)

        if not ready:
            raise TimeoutError(f"MCP server did not respond within {timeout}s")

        line = self.process.stdout.readline()
        if not line:
            if self.process.poll() is not None:
                raise RuntimeError(f"MCP server terminated (exit code: {self.process.returncode})")
            raise RuntimeError("MCP server closed connection")

        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from MCP: {line[:100]}") from e

    def stop(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.stdout.close()
                self.process.stderr.close()
            except Exception:
                pass  # Already closed

            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception as e:
                print(f"Warning: Error stopping MCP server: {e}")
            finally:
                self.process = None
