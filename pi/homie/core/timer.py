"""Timer management with callbacks."""

import threading
import time
import uuid


class TimerManager:
    """Manages in-memory timers with callbacks."""

    # Tool definitions for Anthropic API
    TOOLS = [
        {
            "name": "set_timer",
            "description": "Set a timer that will remind the user after a specified duration. Use for short-term reminders like 'remind me in 30 minutes to take out the cake'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Duration in seconds until the timer fires"
                    },
                    "message": {
                        "type": "string",
                        "description": "The reminder message to speak when the timer fires"
                    }
                },
                "required": ["seconds", "message"]
            }
        },
        {
            "name": "list_timers",
            "description": "List all active timers with their remaining time and messages.",
            "input_schema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "cancel_timer",
            "description": "Cancel an active timer by its ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timer_id": {
                        "type": "string",
                        "description": "The ID of the timer to cancel"
                    }
                },
                "required": ["timer_id"]
            }
        }
    ]

    def __init__(self, on_fire_callback):
        """Initialize timer manager.

        Args:
            on_fire_callback: Function to call when timer fires.
                              Receives (message: str) as argument.
        """
        self.timers = {}  # id -> {due_at, message, timer}
        self.on_fire = on_fire_callback
        self.lock = threading.Lock()

    def set_timer(self, seconds: int, message: str) -> dict:
        """Set a new timer."""
        timer_id = str(uuid.uuid4())[:8]

        def on_timer_fire():
            with self.lock:
                if timer_id in self.timers:
                    del self.timers[timer_id]
            self.on_fire(message)

        t = threading.Timer(seconds, on_timer_fire)
        t.start()

        with self.lock:
            self.timers[timer_id] = {
                "id": timer_id,
                "due_at": time.time() + seconds,
                "message": message,
                "timer": t
            }

        return {
            "success": True,
            "timer_id": timer_id,
            "seconds": seconds,
            "message": message
        }

    def list_timers(self) -> dict:
        """List all active timers."""
        now = time.time()
        with self.lock:
            timers = [{
                "id": t["id"],
                "message": t["message"],
                "seconds_remaining": max(0, int(t["due_at"] - now))
            } for t in self.timers.values()]

        return {
            "timers": timers,
            "count": len(timers)
        }

    def cancel_timer(self, timer_id: str) -> dict:
        """Cancel an active timer."""
        with self.lock:
            if timer_id in self.timers:
                self.timers[timer_id]["timer"].cancel()
                del self.timers[timer_id]
                return {"success": True, "timer_id": timer_id}
            else:
                return {"success": False, "error": f"Timer {timer_id} not found"}

    def cancel_all(self):
        """Cancel all timers (for cleanup)."""
        with self.lock:
            for timer_data in self.timers.values():
                timer_data["timer"].cancel()
            self.timers.clear()

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a timer tool by name."""
        if tool_name == "set_timer":
            return self.set_timer(arguments["seconds"], arguments["message"])
        elif tool_name == "list_timers":
            return self.list_timers()
        elif tool_name == "cancel_timer":
            return self.cancel_timer(arguments["timer_id"])
        else:
            return {"error": f"Unknown timer tool: {tool_name}"}

    @classmethod
    def get_tool_names(cls) -> list[str]:
        """Get list of timer tool names."""
        return [t["name"] for t in cls.TOOLS]
