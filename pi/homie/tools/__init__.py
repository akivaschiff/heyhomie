"""Tool registry — the full durable tool set the brain orchestrates over."""

from homie.tools import (
    kiosk,
    memory,
    recipes,
    reminders,
    schedule,
    shabbat,
    shopping_list,
    smart_home,
    timers,
    volume,
)


def all_tools():
    return [
        *shopping_list.TOOLS,
        *timers.TOOLS,
        *reminders.TOOLS,
        *recipes.TOOLS,
        *memory.TOOLS,
        *shabbat.TOOLS,
        *smart_home.TOOLS,
        *schedule.TOOLS,
        *volume.TOOLS,
        *kiosk.TOOLS,
    ]
