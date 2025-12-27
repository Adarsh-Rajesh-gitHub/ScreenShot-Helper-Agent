import { routeAgentRequest } from "agents";
import { AIChatAgent } from "agents/ai-chat-agent";

import {
  streamText,
  stepCountIs,
  createUIMessageStream,
  convertToModelMessages,
  createUIMessageStreamResponse,
  type StreamTextOnFinishCallback,
} from "ai";

import { createWorkersAI } from "workers-ai-provider";
import { processToolCalls, cleanupMessages } from "./utils";
import { tools, executions } from "./tools";
 //import { env } from "cloudflare:workers";

// Cloudflare AI Gateway
// const openai = createOpenAI({
//   apiKey: env.OPENAI_API_KEY,
//   baseURL: env.GATEWAY_BASE_URL,
// });

/**
 * Chat Agent implementation that handles real-time AI chat interactions
 */
export class Chat extends AIChatAgent<Env> {
  /**
   * Handles incoming chat messages and manages the response stream
   */
  
async onChatMessage(
  onFinish: StreamTextOnFinishCallback<any>,
  _options?: { abortSignal?: AbortSignal }
) {

    const workersai = createWorkersAI({ binding: this.env.AI });
    const model = workersai("@cf/deepseek-ai/deepseek-r1-distill-qwen-32b");
    
    // const mcpConnection = await this.mcp.connect(
    //   "https://path-to-mcp-server/sse"
    // );

    // Collect all tools, including MCP tools
      const allTools = { ...tools };

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        // Clean up incomplete tool calls to prevent API errors
        const cleanedMessages = cleanupMessages(this.messages);

        // Process any pending tool calls from previous messages
        // This handles human-in-the-loop confirmations for tools
        const processedMessages = await processToolCalls({
          messages: cleanedMessages,
          dataStream: writer,
          tools: allTools,
          executions
        });

        const result = streamText({
          system: `You are a helpful assistant. Respond normally like a chat bot. Be concise.
`,

          messages: convertToModelMessages(processedMessages),
          model,
          tools: allTools,
          // Type boundary: streamText expects specific tool types, but base class uses ToolSet
          // This is safe because our tools satisfy ToolSet interface (verified by 'satisfies' in tools.ts)
          onFinish: onFinish as unknown as StreamTextOnFinishCallback<
            typeof allTools
          >,
          stopWhen: stepCountIs(10)
        });

        writer.merge(result.toUIMessageStream());
      }
    });

    return createUIMessageStreamResponse({ stream });
  }
}

/**
 * Worker entry point that routes incoming requests to the appropriate handler
 */
export default {
  
  async fetch(request: Request, env: Env, _ctx: ExecutionContext) {
    const url = new URL(request.url);
    function uint8ToBase64(bytes: Uint8Array): string {
      let binary = "";
      const chunkSize = 0x8000; // 32KB chunks to avoid stack/memory issues
      for (let i = 0; i < bytes.length; i += chunkSize) {
        // biome-ignore lint/suspicious/noExplicitAny: fine for MVP
        binary += String.fromCharCode(...(bytes.subarray(i, i + chunkSize) as any));
      }
      return btoa(binary);
    }
    if (url.pathname === "/check-ai-binding") {
      return Response.json({ success: !!env.AI });
    }
    if (url.pathname === "/capture" && request.method === "POST") {
      const form = await request.formData();

      const goalRaw = form.get("goal");
      const imageRaw = form.get("image");

      // Validate goal first (prevents calling .trim() on non-string)
      if (typeof goalRaw !== "string" || goalRaw.trim().split(/\s+/).length < 2) {
        return Response.json(
          { ok: false, error: "Goal must be at least 2 words." },
          { status: 400 }
        );
      }
      const goal = goalRaw.trim();

      // Validate image next (prevents arrayBuffer() on non-File)
      if (!(imageRaw instanceof File)) {
        return Response.json(
          { ok: false, error: "Missing image file." },
          { status: 400 }
        );
      }
      const image = imageRaw;

      const allowed = new Set(["image/png", "image/jpeg"]);
      if (!allowed.has(image.type)) {
        return Response.json(
          { ok: false, error: "Only PNG/JPG allowed." },
          { status: 415 }
        );
      }

      const maxBytes = 5 * 1024 * 1024;
      if (image.size > maxBytes) {
        return Response.json(
          { ok: false, error: "File too large (max 5MB)." },
          { status: 413 }
        );
      }

      // Build data URL after validation
      const bytes = new Uint8Array(await image.arrayBuffer());
      const base64 = uint8ToBase64(bytes);
      const dataUrl = `data:${image.type};base64,${base64}`;

      // Pick model from env (runtime-configurable). Keep typing flexible for MVP.
      const modelId = (env.VISION_MODEL_ID || "@cf/meta/llama-3.2-11b-vision-instruct") as any;
      const workersai = createWorkersAI({ binding: env.AI });
      const visionModel = workersai(modelId);

      const system =
        "Return ONLY valid JSON with keys: " +
        "screen_summary, ui_elements, steps, confidence, need_new_screenshot, expected_next_screen.";

      const userPrompt = `Goal: ${goal}\nInclude >=6 ui_elements and >=4 steps.`;

      // One-call brain + strict JSON parse with one retry
      let raw: string;
      {
        const r = streamText({
          model: visionModel,
          system,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: userPrompt },
                { type: "image", image: dataUrl }
              ]
            }
          ],
          stopWhen: stepCountIs(5)
        });
        raw = await r.text;
      }

      let brain: any;
      try {
        brain = JSON.parse(raw);
      } catch {
        const r2 = streamText({
          model: visionModel,
          system: system + "\nCRITICAL: output parseable JSON only.",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: userPrompt },
                { type: "image", image: dataUrl }
              ]
            }
          ],
          stopWhen: stepCountIs(5)
        });
        const raw2 = await r2.text;

        try {
          brain = JSON.parse(raw2);
        } catch {
          console.error("Non-JSON model output:", raw2.slice(0, 1200));
          return Response.json(
            { ok: false, error: "Model returned invalid JSON." },
            { status: 502 }
          );
        }
      }

      return Response.json({
        ok: true,
        received: {
          filename: image.name,
          type: image.type,
          size: image.size,
          goal
        },
        brain
      });
    }

    return (
      // Route the request to our agent or return 404 if not found
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
