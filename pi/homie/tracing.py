"""Observability seam. The brain emits one trace per user turn, with a generation
per Claude call and a span per tool execution. NoopTracer keeps the core
framework-free (tests and the Mac harness need no Langfuse); LangfuseTracer wires
the real backend when credentials are present.
"""

import os


class _NoopRec:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def result(self, **kwargs):
        pass


class NoopTracer:
    enabled = False

    def turn(self, user_text):
        return _NoopRec()

    def generation(self, messages, model, params, system=None, tools=None):
        return _NoopRec()

    def tool(self, name, tool_input):
        return _NoopRec()

    def stt(self, audio_bytes=0):
        return _NoopRec()

    def tts(self, text=""):
        return _NoopRec()

    def flush(self):
        pass


class _LangfuseRec:
    def __init__(self, cm):
        self._cm = cm
        self._obs = None

    def __enter__(self):
        self._obs = self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def result(self, output=None, usage=None):
        update = {}
        if output is not None:
            update["output"] = output
        if usage is not None:
            update["usage_details"] = usage
        if update and self._obs is not None:
            self._obs.update(**update)


class LangfuseTracer:
    enabled = True

    def __init__(self, client):
        self.client = client

    def turn(self, user_text):
        return _LangfuseRec(
            self.client.start_as_current_observation(
                name="turn", as_type="span", input=user_text
            )
        )

    def generation(self, messages, model, params, system=None, tools=None):
        return _LangfuseRec(
            self.client.start_as_current_observation(
                name="claude",
                as_type="generation",
                input={"system": system, "messages": messages},
                model=model,
                model_parameters=params,
                metadata={"tools": tools},
            )
        )

    def tool(self, name, tool_input):
        return _LangfuseRec(
            self.client.start_as_current_observation(
                name=name, as_type="tool", input=tool_input
            )
        )

    def stt(self, audio_bytes=0):
        return _LangfuseRec(
            self.client.start_as_current_observation(
                name="stt", as_type="span", metadata={"audio_bytes": audio_bytes}
            )
        )

    def tts(self, text=""):
        return _LangfuseRec(
            self.client.start_as_current_observation(
                name="tts", as_type="span", input=text
            )
        )

    def flush(self):
        self.client.flush()


def build_tracer():
    """LangfuseTracer if credentials are set and the SDK is importable, else a
    no-op. Tracing is observability — a missing SDK or bad config must never take
    the assistant down."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return NoopTracer()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
        return LangfuseTracer(client)
    except Exception as exc:
        print(f"⚠️  Langfuse tracing disabled: {exc}")
        return NoopTracer()
