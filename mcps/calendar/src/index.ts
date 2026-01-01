import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { google } from "googleapis";
import { z } from "zod";

const CREDENTIALS_PATH = process.env.GOOGLE_SERVICE_ACCOUNT_PATH || "./service-account.json";
const DEFAULT_CALENDAR_ID = process.env.DEFAULT_CALENDAR_ID || "primary";

async function getCalendarClient() {
  const auth = new google.auth.GoogleAuth({
    keyFile: CREDENTIALS_PATH,
    scopes: ["https://www.googleapis.com/auth/calendar.readonly"],
  });
  return google.calendar({ version: "v3", auth });
}

const server = new McpServer({
  name: "homie-calendar",
  version: "1.0.0",
});

server.tool(
  "list_calendars",
  "List all calendars available to this assistant. Returns the user's configured calendar.",
  {},
  async () => {
    const calendar = await getCalendarClient();

    // Try to get the default calendar directly
    const calendars = [];

    try {
      const cal = await calendar.calendars.get({ calendarId: DEFAULT_CALENDAR_ID });
      calendars.push({
        id: DEFAULT_CALENDAR_ID,
        name: cal.data.summary || DEFAULT_CALENDAR_ID,
        description: cal.data.description || "User's primary calendar",
        primary: true,
      });
    } catch (error) {
      // If default calendar fails, try listing from calendarList
      const res = await calendar.calendarList.list();
      const items = res.data.items?.map((c) => ({
        id: c.id,
        name: c.summary,
        description: c.description,
        primary: c.primary,
      })) || [];
      calendars.push(...items);
    }

    return { content: [{ type: "text", text: JSON.stringify(calendars, null, 2) }] };
  }
);

server.tool(
  "list_events",
  "List events from the user's calendar. If no calendarId is specified, uses the user's primary calendar.",
  {
    calendarId: z.string().optional().describe("Calendar ID (defaults to user's primary calendar)"),
    maxResults: z.number().optional().default(10).describe("Maximum number of events to return"),
    timeMin: z.string().optional().describe("Start time (ISO 8601 format). Defaults to now."),
    timeMax: z.string().optional().describe("End time (ISO 8601 format)"),
  },
  async ({ calendarId, maxResults, timeMin, timeMax }) => {
    const calendar = await getCalendarClient();
    const res = await calendar.events.list({
      calendarId: calendarId || DEFAULT_CALENDAR_ID,
      maxResults,
      timeMin: timeMin || new Date().toISOString(),
      timeMax,
      singleEvents: true,
      orderBy: "startTime",
    });
    const events = res.data.items?.map((e) => ({
      id: e.id,
      summary: e.summary,
      description: e.description,
      start: e.start?.dateTime || e.start?.date,
      end: e.end?.dateTime || e.end?.date,
      location: e.location,
    })) || [];
    return { content: [{ type: "text", text: JSON.stringify(events, null, 2) }] };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
