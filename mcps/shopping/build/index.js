import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { google } from "googleapis";
import { z } from "zod";
// Config
const CREDENTIALS_PATH = process.env.GOOGLE_SERVICE_ACCOUNT_PATH || "./service-account.json";
const SPREADSHEET_ID = process.env.PANTRY_SHEET_ID;
if (!SPREADSHEET_ID)
    throw new Error("PANTRY_SHEET_ID environment variable is required");
const ITEMS_SHEET = "Items";
const LOG_SHEET = "Log";
// Sheets client
async function getSheetsClient() {
    const auth = new google.auth.GoogleAuth({
        keyFile: CREDENTIALS_PATH,
        scopes: ["https://www.googleapis.com/auth/spreadsheets"],
    });
    return google.sheets({ version: "v4", auth });
}
// Helper: get all items
async function getItems() {
    const sheets = await getSheetsClient();
    const res = await sheets.spreadsheets.values.get({
        spreadsheetId: SPREADSHEET_ID,
        range: `${ITEMS_SHEET}!A2:F`,
    });
    const rows = res.data.values || [];
    return rows.map((row, idx) => ({
        row: idx + 2,
        name: row[0] || "",
        unit: row[1] || null,
        quantity: row[2] ? Number(row[2]) : null,
        level: row[3] || "good",
        par: row[4] ? Number(row[4]) : null,
        updatedAt: row[5] || "",
    }));
}
// Helper: find item by name (case-insensitive)
async function findItem(name) {
    const items = await getItems();
    return items.find((i) => i.name.toLowerCase() === name.toLowerCase()) || null;
}
// Helper: update item row
async function updateItem(row, data) {
    const sheets = await getSheetsClient();
    const item = (await getItems()).find((i) => i.row === row);
    if (!item)
        throw new Error("Item not found");
    const updated = { ...item, ...data, updatedAt: new Date().toISOString() };
    await sheets.spreadsheets.values.update({
        spreadsheetId: SPREADSHEET_ID,
        range: `${ITEMS_SHEET}!A${row}:F${row}`,
        valueInputOption: "RAW",
        requestBody: {
            values: [[updated.name, updated.unit || "", updated.quantity ?? "", updated.level, updated.par ?? "", updated.updatedAt]],
        },
    });
}
// Helper: append new item
async function appendItem(name, unit, quantity, level, par) {
    const sheets = await getSheetsClient();
    await sheets.spreadsheets.values.append({
        spreadsheetId: SPREADSHEET_ID,
        range: `${ITEMS_SHEET}!A:F`,
        valueInputOption: "RAW",
        requestBody: {
            values: [[name, unit || "", quantity ?? "", level, par ?? "", new Date().toISOString()]],
        },
    });
}
// Helper: log action
async function logAction(item, action, quantityChange, note, rawInput) {
    const sheets = await getSheetsClient();
    await sheets.spreadsheets.values.append({
        spreadsheetId: SPREADSHEET_ID,
        range: `${LOG_SHEET}!A:F`,
        valueInputOption: "RAW",
        requestBody: {
            values: [[new Date().toISOString(), item, action, quantityChange ?? "", note, rawInput]],
        },
    });
}
// Helper: derive level from quantity and par
function deriveLevel(quantity, par) {
    if (quantity === null || par === null)
        return "good";
    if (quantity === 0)
        return "out";
    if (quantity <= par * 0.25)
        return "low";
    if (quantity >= par)
        return "stocked";
    return "good";
}
// MCP Server
const server = new McpServer({
    name: "homie-shopping",
    version: "1.0.0",
});
server.tool("list_pantry", "List all items in the pantry", {}, async () => {
    const items = await getItems();
    return { content: [{ type: "text", text: JSON.stringify(items, null, 2) }] };
});
server.tool("get_item", "Get a specific pantry item by name", { name: z.string().describe("Item name") }, async ({ name }) => {
    const item = await findItem(name);
    if (!item)
        return { content: [{ type: "text", text: `Item "${name}" not found` }] };
    return { content: [{ type: "text", text: JSON.stringify(item, null, 2) }] };
});
server.tool("add_item", "Add a new item to the pantry", {
    name: z.string().describe("Item name"),
    unit: z.string().optional().describe("Unit (cans, bottles, bags, etc.)"),
    quantity: z.number().optional().describe("Current quantity (omit for fuzzy tracking)"),
    level: z.enum(["out", "low", "good", "stocked"]).optional().default("good").describe("Current level"),
    par: z.number().optional().describe("Target quantity to maintain"),
    rawInput: z.string().describe("Original voice input that triggered this action"),
}, async ({ name, unit, quantity, level, par, rawInput }) => {
    const existing = await findItem(name);
    if (existing)
        return { content: [{ type: "text", text: `Item "${name}" already exists` }] };
    await appendItem(name, unit || null, quantity ?? null, level, par ?? null);
    await logAction(name, "created", quantity ?? null, `Added to pantry`, rawInput);
    return { content: [{ type: "text", text: `Added "${name}" to pantry` }] };
});
server.tool("use_item", "Record using/consuming an item (decrements quantity or lowers level)", {
    name: z.string().describe("Item name"),
    quantity: z.number().optional().describe("Amount used (for exact tracking)"),
    level: z.enum(["out", "low", "good", "stocked"]).optional().describe("New level (for fuzzy tracking)"),
    rawInput: z.string().describe("Original voice input that triggered this action"),
}, async ({ name, quantity, level, rawInput }) => {
    const item = await findItem(name);
    if (!item)
        return { content: [{ type: "text", text: `Item "${name}" not found` }] };
    if (item.quantity !== null && quantity) {
        // Exact tracking
        const newQty = Math.max(0, item.quantity - quantity);
        const newLevel = deriveLevel(newQty, item.par);
        await updateItem(item.row, { quantity: newQty, level: newLevel });
        await logAction(name, "used", -quantity, `${item.quantity} → ${newQty}`, rawInput);
        return { content: [{ type: "text", text: `Used ${quantity} ${item.unit || "units"} of "${name}". Now have ${newQty}.` }] };
    }
    else if (level) {
        // Fuzzy tracking
        await updateItem(item.row, { level });
        await logAction(name, "used", null, `Level: ${item.level} → ${level}`, rawInput);
        return { content: [{ type: "text", text: `Updated "${name}" to ${level}` }] };
    }
    else {
        return { content: [{ type: "text", text: `Please specify quantity or level for "${name}"` }] };
    }
});
server.tool("restock_item", "Record restocking an item (adds quantity or raises level)", {
    name: z.string().describe("Item name"),
    quantity: z.number().optional().describe("Amount added (for exact tracking)"),
    level: z.enum(["out", "low", "good", "stocked"]).optional().describe("New level (for fuzzy tracking)"),
    rawInput: z.string().describe("Original voice input that triggered this action"),
}, async ({ name, quantity, level, rawInput }) => {
    const item = await findItem(name);
    if (!item)
        return { content: [{ type: "text", text: `Item "${name}" not found. Use add_item first.` }] };
    if (item.quantity !== null && quantity) {
        // Exact tracking
        const newQty = item.quantity + quantity;
        const newLevel = deriveLevel(newQty, item.par);
        await updateItem(item.row, { quantity: newQty, level: newLevel });
        await logAction(name, "restocked", quantity, `${item.quantity} → ${newQty}`, rawInput);
        return { content: [{ type: "text", text: `Restocked ${quantity} ${item.unit || "units"} of "${name}". Now have ${newQty}.` }] };
    }
    else if (level) {
        // Fuzzy tracking
        await updateItem(item.row, { level });
        await logAction(name, "restocked", null, `Level: ${item.level} → ${level}`, rawInput);
        return { content: [{ type: "text", text: `Updated "${name}" to ${level}` }] };
    }
    else {
        return { content: [{ type: "text", text: `Please specify quantity or level for "${name}"` }] };
    }
});
server.tool("get_low_items", "Get all items that are low or out of stock", {}, async () => {
    const items = await getItems();
    const low = items.filter((i) => i.level === "out" || i.level === "low");
    if (low.length === 0)
        return { content: [{ type: "text", text: "All items are stocked!" }] };
    return { content: [{ type: "text", text: JSON.stringify(low, null, 2) }] };
});
server.tool("set_item_out", "Quickly mark an item as out of stock", {
    name: z.string().describe("Item name"),
    rawInput: z.string().describe("Original voice input that triggered this action"),
}, async ({ name, rawInput }) => {
    const item = await findItem(name);
    if (!item)
        return { content: [{ type: "text", text: `Item "${name}" not found` }] };
    await updateItem(item.row, { quantity: item.quantity !== null ? 0 : null, level: "out" });
    await logAction(name, "used", null, `Marked as out`, rawInput);
    return { content: [{ type: "text", text: `Marked "${name}" as out of stock` }] };
});
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
}
main().catch(console.error);
