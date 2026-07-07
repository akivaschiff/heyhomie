Home Assistant — Specification
1. Purpose
A voice-first kitchen assistant with a remote text channel and a local test channel. Capabilities: a supermarket list, kitchen tools (timers, reminders, recipes), an ambient memory store, and a Shabbat-times display. Conversational, with follow-ups that don't require re-waking within a short listening window, and recent context retained for ~1–2 hours.
2. Channels
ChannelRoleActivationDevice-bound renderingVoice (Pi)Primary, kitchenPorcupine wake word, EnglishTablet + kitchen speakerTelegramRemote, when awayInbound messageInline chatMacTest / dev harnessHotkey / push-to-talk (no Porcupine)Mac's own screen + speaker, or no-op
Decisions:

All three are executors over one shared core. Same tools, same shared state, no per-channel brain.
The Mac channel is test-first. Device-bound tools resolve to the Mac itself and may be minimal or no-op; it is not owed full fidelity.
The Mac channel doubles as the string-driven harness: the whole assistant must be drivable from typed input with no audio in the stack. This is a hard portability requirement (see §7).

3. Render-capability model
Device-bound output is a per-channel capability, not a fixed assumption. A tool that wants to "show" something asks the channel what surface it has:

Pi → tablet + speaker
Mac → own screen + speaker (or no-op)
Telegram → inline chat
Headless → none

Tools never assume a tablet exists.
4. Architecture decisions (load-bearing)

Deepgram owns voice orchestration — STT, turn-taking, barge-in, TTS. We do not build the voice loop.
The assistant is a set of tools. The model handles orchestration, clarifying questions, and sequencing. No hand-built dialog state machine.
Disambiguation is native to the model (e.g. "which apples?"). No custom slot-filling.
Tools are the durable layer; channels are executors.
Execution is per-channel; state is shared. Voice/home-touching tools execute on the Pi (home-local). Telegram executes remotely. All persistent data lives in one shared store reachable from home and remote — the Pi is never the system of record.
No confirmation on actions. Every write is additive and cheaply reversible, so all actions execute immediately. (No person-to-person messaging exists in this spec, which is the only thing that would have warranted a confirm.)
Two distinct windows: short listening window (re-wake threshold) vs. ~1–2h rolling conversation context (memory horizon). The context horizon is retained across wake cycles.

5. Tools
Each tool lists its success criteria as observable behavior.
5.1 list_add
Add an item to the supermarket list (one list, for now).

Adds the named item to the list immediately, no confirmation.
Ambiguous items trigger a model clarification before adding ("apples" → "which?" → "green" → adds "green apples"). The clarification is conversational, not a separate step the spec defines.
Phrasings like "we finished the garbage bags" / "I need more apples" are understood as add intents.
Success = item appears on the shared list and is visible from any channel.

5.2 list_remove
Remove or correct an item.

"remove the last one," "not red, green" both work.
Success = the item is gone / corrected on the shared list.

5.3 list_show
Render the current list.

On a channel with a screen → displays the list on that screen (tablet / Mac).
"send it to Telegram" → the list is delivered into the Telegram chat.
Success = the full current list is presented on the requested surface.

5.4 timer_set
Set one or more concurrent named timers.

Multiple timers run at once, each independently named ("cake in the oven" 30 min, "flip pancakes" 30 s).
Fires on the kitchen speaker.
Success = each timer rings at its set duration with its name announced; multiple can be pending simultaneously without collision.

5.5 reminder_set
Set a reminder at a clock time or after a duration.

"reminder for 17:20, prepare dough" and duration-based both work.
Delivered to the kitchen speaker only. (Accepted limitation: missed if no one is in the kitchen.)
Success = the reminder announces on the kitchen speaker at the specified time.

5.6 recipe_load
Resolve a recipe, process it, and display it.

Two sources:

Curated — "chicken from [source]" resolves against your curated list of recipe links.
Search — "find me a recipe for X" performs a web search.


The recipe is fetched, processed, and shown on the channel's screen (tablet / Mac). Telegram receives it inline.
After load, the recipe enters conversation context for the retention horizon.
Success = the correct recipe is displayed on the available surface and is available for follow-up questions afterward.

5.7 memory_save / memory_query
Free-form fact store.

Save: "the spare key is in the shed" → stored.
Query: "where's the spare key?" → returns the stored fact.
Channel-agnostic; works identically from any channel against shared state.
Success = a saved fact is retrievable later, including from a different channel.

5.8 shabbat_mode
Display Shabbat times.

Shows an HTML page of Shabbat times on the tablet. Display only — no action gating, no suppression, no electrical blocking.
Invoked on request (manual). Auto-activation at candle-lighting is explicitly out for v1.
Success = the Shabbat-times page renders on the tablet when requested.

Model-handled, not tools
These are handled by the model reasoning over the recipe held in context — no tool, no separate state:

Step navigation — "next," "repeat," "how much flour."
Scaling — "make it for 6." Re-rendered via the existing show-on-screen path if a display is present.
Unit conversion — defaults to metric.
Substitutions — "out of buttermilk, what instead."

These work as long as the referenced recipe is within the ~1–2h context horizon, even across re-wakes.
6. Flows that must work

Easy add — "we finished the garbage bags" → garbage bags on the list, short confirmation.
Add with disambiguation — "I need more apples" → "which apples?" → "green" → green apples added.
Show / send list — "show the list" → on screen; "send it to Telegram" → in chat.
Concurrent named timers — set "cake in the oven" 30 min and "flip pancakes" 30 s; both pending; each rings by name.
Clock reminder — "reminder for 17:20, prepare dough" → fires on kitchen speaker at 17:20.
Curated recipe — "show me chicken from [source]" → resolved from curated links → displayed.
Search recipe — "find me a recipe for [dish]" → web search → displayed.
Hands-free across re-wake — load cookie recipe; minutes later, re-wake and ask "how much flour?" → answered from retained context without reloading.
Scale + re-display — "make it for 6" → model rescales the loaded recipe → re-rendered to screen.
Ambient memory round-trip — save a fact on voice, recall it later from Telegram.
Cross-channel list — add an item on the Pi; read the same list over Telegram while out.
Shabbat times — request Shabbat mode → times page on tablet.
Mac harness — every flow above runs on the Mac via typed input (audio optional), device-bound steps resolving to the Mac or no-op.

7. Portability requirement (hard criterion)
The program must run on the Mac with wake word removed (hotkey/push-to-talk substituted) and device-bound tools resolving to the Mac or no-op. Equivalently: the core must be drivable end-to-end from plain typed input with no audio anywhere in the stack. If any flow can't be exercised this way, voice has leaked into the core and must be pulled back out.
8. Out of scope (v1)

Expenses; home / garden / pool logging.
Jewish-practice features beyond the Shabbat-times page (no zmanim, no action gating).
Routines / briefings / scenes.
Multiple lists (one supermarket list only).
Hebrew voice (English only).
Whole-house / multi-room voice (single kitchen mic).
Person-to-person messaging.
Any destructive or irreversible action.