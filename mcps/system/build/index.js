import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { exec } from "child_process";
import { promisify } from "util";
const execAsync = promisify(exec);
const server = new McpServer({
    name: "homie-system",
    version: "1.0.0",
});
// Get current date and time
server.tool("get_datetime", "Get the current date and time. Use this when the user asks about the time, date, or day of the week.", {
    timezone: z.string().optional().describe("Timezone (e.g., 'Asia/Jerusalem', 'America/New_York', 'Europe/London'). If not specified, uses system timezone."),
}, async ({ timezone }) => {
    const now = new Date();
    // Use system timezone if not specified
    const effectiveTimezone = timezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
    const options = {
        timeZone: effectiveTimezone,
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
        timeZoneName: "short",
    };
    const formatted = now.toLocaleString("en-US", options);
    return {
        content: [{
                type: "text",
                text: JSON.stringify({
                    formatted,
                    iso: now.toISOString(),
                    timestamp: now.getTime(),
                    timezone,
                }, null, 2),
            }],
    };
});
// Set volume
server.tool("set_volume", "Set the audio output volume. Use this when the user asks to change, increase, decrease, or mute the volume.", {
    volume: z.number().min(0).max(100).describe("Volume level (0-100, where 0 is mute and 100 is maximum)"),
}, async ({ volume }) => {
    try {
        // Use amixer to set volume on Linux (Raspberry Pi)
        const { stdout, stderr } = await execAsync(`amixer sset 'Master' ${volume}%`);
        return {
            content: [{
                    type: "text",
                    text: JSON.stringify({
                        success: true,
                        volume,
                        message: `Volume set to ${volume}%`,
                    }, null, 2),
                }],
        };
    }
    catch (error) {
        return {
            content: [{
                    type: "text",
                    text: JSON.stringify({
                        success: false,
                        error: error instanceof Error ? error.message : String(error),
                    }, null, 2),
                }],
        };
    }
});
// Get current volume
server.tool("get_volume", "Get the current audio output volume level. Use this when the user asks what the current volume is.", {}, async () => {
    try {
        // Parse amixer output to get current volume
        const { stdout } = await execAsync("amixer sget 'Master'");
        // Extract volume percentage from output like: [60%]
        const match = stdout.match(/\[(\d+)%\]/);
        const volume = match ? parseInt(match[1]) : null;
        return {
            content: [{
                    type: "text",
                    text: JSON.stringify({
                        success: true,
                        volume,
                        message: volume !== null ? `Current volume is ${volume}%` : "Could not determine volume",
                    }, null, 2),
                }],
        };
    }
    catch (error) {
        return {
            content: [{
                    type: "text",
                    text: JSON.stringify({
                        success: false,
                        error: error instanceof Error ? error.message : String(error),
                    }, null, 2),
                }],
        };
    }
});
// Update and reboot
server.tool("update_and_reboot", "Update Homie from git and reboot the system. Use this when the user asks to update, upgrade, or get the latest version of Homie.", {}, async () => {
    try {
        // Determine the homie directory path
        const homieDir = process.env.HOMIE_DIR || "/home/akiva/heyhomie";
        // Run git pull
        const { stdout, stderr } = await execAsync(`cd ${homieDir} && git pull`);
        // Check if there were any errors
        const hasErrors = stderr && !stderr.includes("Already up to date");
        // Schedule reboot in 5 seconds to allow response to be sent
        execAsync("sudo shutdown -r +0.1").catch(() => { });
        return {
            content: [{
                    type: "text",
                    text: JSON.stringify({
                        success: true,
                        message: "Updating and rebooting in 5 seconds",
                        gitOutput: stdout,
                        errors: hasErrors ? stderr : null,
                    }, null, 2),
                }],
        };
    }
    catch (error) {
        return {
            content: [{
                    type: "text",
                    text: JSON.stringify({
                        success: false,
                        error: error instanceof Error ? error.message : String(error),
                    }, null, 2),
                }],
        };
    }
});
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
}
main().catch(console.error);
