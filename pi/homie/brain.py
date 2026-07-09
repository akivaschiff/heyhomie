"""The brain — a channel-agnostic Claude tool loop.

The model handles orchestration, clarifying questions, disambiguation and
sequencing. There is no hand-built dialog state machine and no per-channel logic
here: feed it user text, it runs tools against shared state, and the active
channel renders the result. This is the layer the portability requirement
protects — no audio, no STT/TTS, drivable from plain typed input.
"""

import json

from homie.context import ConversationContext
from homie.tracing import NoopTracer

MAX_TOOL_ITERATIONS = 8


def build_system_prompt(channel, clock) -> str:
    now = clock.now().strftime("%A, %B %d, %Y at %H:%M %Z")
    surface = channel.surface
    caps = []
    if surface.has_screen:
        caps.append("a screen you can show pages on")
    if surface.has_speaker:
        caps.append("a speaker")
    if surface.is_chat:
        caps.append("an inline chat")
    surface_desc = ", ".join(caps) if caps else "no display (audio/text only)"

    return f"""You are Homie, a voice-first kitchen assistant.

Current date and time: {now}

You run over a set of tools and shared state (a supermarket list, kitchen timers
and reminders, recipes, an ambient memory store, and Shabbat times). The state is
shared across every channel — the list you add to on voice is the same list read
over Telegram.

This channel has: {surface_desc}.
- To present a list, recipe, or Shabbat times, call the tool that shows it; it
  renders to whatever surface exists here. Never assume a tablet.
- Adds and removes happen immediately, no confirmation. Every action is additive
  and cheaply reversible.
- For ambiguous items ("apples"), ask one short clarifying question before acting.
- Understand intents like "we finished the garbage bags" or "I need more apples"
  as add-to-list.

{_style_block(surface)}"""


def _style_block(surface) -> str:
    if surface.has_speaker:
        return """Your replies are spoken aloud through text-to-speech. This means:
- Plain conversational speech ONLY. Never use markdown: no asterisks, no **bold**,
  no bullet points, no numbered lists, no headings, no emoji. They get read out
  literally ("star star").
- Keep it to 1-2 short sentences, under 30 words total — even for overviews and
  lists. Weave items into a sentence ("you've got milk, eggs and bread") and stop;
  offer to go deeper rather than covering everything.
- Tool results arrive as JSON — never read JSON back; answer naturally from it."""
    return "Keep replies short and conversational. 1-2 sentences."


class Brain:
    def __init__(self, anthropic_client, tools, tool_context, config, clock, tracer=None):
        self.client = anthropic_client
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        self.ctx = tool_context
        self.config = config
        self.clock = clock
        self.tracer = tracer or NoopTracer()
        self.conversation = ConversationContext(config.context_window_seconds, clock)

    def _tool_schemas(self):
        return [t.anthropic_schema() for t in self.tools]

    def _run_tool(self, name: str, tool_input: dict) -> str:
        tool = self.tools_by_name.get(name)
        with self.tracer.tool(name, tool_input) as span:
            if tool is None:
                result = json.dumps({"error": f"unknown tool: {name}"})
            else:
                try:
                    result = tool.handler(tool_input, self.ctx)
                except Exception as exc:  # surface failures to the model, don't crash
                    result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            span.result(output=result)
            return result

    def handle(self, user_text: str) -> str:
        """Run one user turn to completion and return the final assistant text.

        Every exit path leaves the conversation ending on an assistant message, so
        the next turn's user message never produces two consecutive user messages
        (which the API rejects)."""
        self.conversation.start_turn(user_text)
        with self.tracer.turn(user_text) as turn:
            try:
                final = self._run_loop()
            except Exception as exc:
                import traceback

                print(f"⚠️  turn failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                final = "Sorry, something went wrong."
                self.conversation.add("assistant", [{"type": "text", "text": final}])
                self.ctx.channel.deliver(final)
                turn.result(output=f"error: {type(exc).__name__}: {exc}")
                self.tracer.flush()
                return final
            turn.result(output=final)
        self.tracer.flush()
        return final

    def _run_loop(self) -> str:
        system = build_system_prompt(self.ctx.channel, self.clock)
        schemas = self._tool_schemas() or None
        params = {"max_tokens": self.config.max_tokens}

        for _ in range(MAX_TOOL_ITERATIONS):
            messages = self.conversation.get()
            with self.tracer.generation(
                messages, self.config.model, params, system=system, tools=schemas
            ) as gen:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=system,
                    messages=messages,
                    tools=schemas,
                )
                gen.result(
                    output=[b.model_dump() for b in response.content],
                    usage={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                )

            self.conversation.add("assistant", [b.model_dump() for b in response.content])

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # No tools requested -> this is the final assistant turn.
            if not tool_results:
                text = "".join(b.text for b in response.content if b.type == "text")
                self.ctx.channel.deliver(text)
                return text

            self.conversation.add("user", tool_results)

        fallback = "Sorry, I got stuck on that one."
        self.conversation.add("assistant", [{"type": "text", "text": fallback}])
        self.ctx.channel.deliver(fallback)
        return fallback
