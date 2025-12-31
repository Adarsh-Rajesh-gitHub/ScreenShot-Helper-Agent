import { routeAgentRequest } from "agents";
import { AIChatAgent } from "agents/ai-chat-agent";
export { SessionMemoryDO } from "./session_memory";
import type { SessionState, Brain } from "./session_memory";

import {
  streamText,
  stepCountIs,
  createUIMessageStream,
  convertToModelMessages,
  createUIMessageStreamResponse,
  type StreamTextOnFinishCallback
} from "ai";

import { createWorkersAI } from "workers-ai-provider";
import { processToolCalls, cleanupMessages } from "./utils";
import { tools, executions } from "./tools";



export class Chat extends AIChatAgent<Env> {
  /**
   * Handles incoming chat messages and manages the response stream
   */

  private async readSessionState(): Promise<SessionState | null> {
    try {
      console.log("[chat] this.name (DO key):", this.name);

      // IMPORTANT: use the agent instance name as the durable-session key
      const id = this.env.SESSION_MEMORY.idFromName(this.name);
      const stub = this.env.SESSION_MEMORY.get(id);

      const r = await stub.fetch("https://do/session", { method: "GET" });
      console.log("[chat] /session fetch ok?:", r.ok, "status:", r.status);
      if (!r.ok) return null;

      const data = (await r.json()) as any;
      console.log(
        "[chat] session has brain?:",
        Boolean(data?.session?.last_result_json),
        "history_len:",
        data?.session?.history?.length ?? 0
      );

      return (data?.session as SessionState) ?? null;
    } catch (e: any) {
      console.log("[chat] readSessionState error:", String(e?.message ?? e));
      return null;
    }
  }

  private formatScreenshotContext(session: SessionState | null): string {
    const brain: Brain | null = session?.last_result_json ?? null;
    if (!brain) {
      return "";
    }

    // Try to use the most recent goal from history if present
    const lastGoal =
      session?.history && session.history.length
        ? session.history[session.history.length - 1]?.goal
        : null;

    const activeGoal = session?.active_goal ?? lastGoal;
    const agentMemo =
      typeof session?.agent_memo === "string" && session.agent_memo.trim().length
        ? session.agent_memo.trim()
        : null;

    const elements = Array.isArray(brain.ui_elements) ? brain.ui_elements : [];
    const steps = Array.isArray(brain.steps) ? brain.steps : [];

    let visionText =
      typeof brain.vision_text === "string" ? brain.vision_text : "";
    if (visionText.length > 2000) visionText = visionText.slice(0, 2000) + "…";

    const elementsLines = elements
      .slice(0, 12)
      .map((e) => `- ${e?.label ?? "?"} [${e?.type ?? "?"}]${e?.hint ? `: ${e.hint}` : ""}`)
      .join("\n");

    const stepsLines = steps
      .slice(0, 8)
      .map((s, i) => `${i + 1}. ${s}`)
      .join("\n");
      let brainJson = "";
      try {
        brainJson = JSON.stringify(brain);
      } catch {
        brainJson = "";
      }
      if (brainJson.length > 3500) brainJson = brainJson.slice(0, 3500) + "…";
          return (
      "\n\n" +
      `If screenshot analysis context is present, you MUST answer with:\n` +
`1) WHERE the target UI element is (left/right/top/bottom + what it looks like)\n` +
`2) Steps to complete the goal (numbered)\n` +
`Only ask for a new screenshot if the element is NOT visible or need_new_screenshot=true.\n` +
      "You have structured analysis from the user's most recent uploaded screenshot. " +
      "Use it together with the user's message to answer. " +
      "If the user asks for something that requires seeing a *new* screen and the context is stale, ask for a new screenshot.\n" +
      "--- SCREENSHOT_ANALYSIS ---\n" +
      `Goal (from upload): ${lastGoal ?? "(unknown)"}\n` +
            `Active goal: ${activeGoal ?? "(unknown)"}\n` +
      `Agent memo: ${agentMemo ?? "(none)"}\n` +
      `Screen summary: ${brain.screen_summary ?? "(none)"}\n` +
      `Confidence: ${typeof brain.confidence === "number" ? brain.confidence : "(n/a)"}\n` +
      `Need new screenshot: ${brain.need_new_screenshot ? "true" : "false"}\n` +
      `Expected next screen: ${brain.expected_next_screen ?? "(unknown)"}\n` +
      "UI elements:\n" +
      (elementsLines || "(none)") +
      "\n\nSteps (suggested by analysis):\n" +
      (stepsLines || "(none)") +
      "\n\nVision text (from screenshot):\n" +
      (visionText || "(unavailable)") +
      "\n\nFull structured JSON:\n" +
      (brainJson || "(unavailable)") +
      "\n--- END_SCREENSHOT_ANALYSIS ---\n"
    );
  }

  async onChatMessage(
    onFinish: StreamTextOnFinishCallback<any>,
    _options?: { abortSignal?: AbortSignal }
  ) {
    const workersai = createWorkersAI({ binding: this.env.AI });
    const model = workersai("@cf/meta/llama-3.1-8b-instruct" as any);
    // const mcpConnection = await this.mcp.connect(   
    //   "https://path-to-mcp-server/sse"
    // );

    // Collect all tools, including MCP tools
const allTools = {};
    // Helper: only allow the local-time tool when the user explicitly asks for time/date.
    // This prevents irrelevant tool calls from hijacking the answer when we already have screenshot context.
    const toolsForUserText = (userText: string) => {
      const wantsTime = /\b(time|timezone|date|clock|today|tomorrow)\b/i.test(userText);
      if (wantsTime) return allTools;

      // Filter out getLocalTime (and any accidental prefixed variant)
      const filtered = Object.fromEntries(
        Object.entries(allTools).filter(([name]) => name !== "getLocalTime" && name !== "tool-getLocalTime")
      );
      return filtered as typeof allTools;
    };

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

        // Load latest screenshot analysis for this agent instance (keyed by this.name)
        const session = await this.readSessionState();
        const screenshotContext = this.formatScreenshotContext(session);

        // Derive a simple user-text string for tool gating / logging
        const lastUserMsg = [...processedMessages].reverse().find((m: any) => m?.role === "user");
        const userText =
          typeof (lastUserMsg as any)?.content === "string"
            ? ((lastUserMsg as any).content as string)
            : JSON.stringify((lastUserMsg as any)?.content ?? "");

        console.log("[chat] screenshotContext chars:", screenshotContext.length);
        console.log("[chat] screenshotContext preview:", screenshotContext.slice(0, 200));
        console.log("[chat] last user text:", userText.slice(0, 200));

        const toolsForTurn = toolsForUserText(userText);

        const result = streamText({
          system:
            `You are a helpful assistant. Be concise. Do not output <think> tags or any XML/HTML-like tags.\n` +
            `When a screenshot analysis context is present, you MUST use it together with the user's message.\n` +
            `Only call tools when they are clearly necessary to answer the user's request.\n` +
            `NEVER call getLocalTime unless the user explicitly asked for the time/date/timezone.\n` +
            `If the user did not specify what they want done, ask a single clarifying question.\n` +
            screenshotContext,

          messages: convertToModelMessages(processedMessages),
          model,
          tools: toolsForTurn,
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
    if (url.pathname === "/check-ai-binding") {
      return Response.json({ success: !!env.AI });
    }
    if (url.pathname === "/agree" && request.method === "POST") {
      try {
        if (!env.AI) {
          return Response.json({ ok: false, error: "Missing env.AI binding" }, { status: 500 });
        }

        const modelId =
          env.VISION_MODEL_ID || "@cf/meta/llama-3.2-11b-vision-instruct";

        await env.AI.run(modelId as any, { prompt: "agree" } as any);

        return Response.json({ ok: true, agreed: true });
      } catch (e: any) {
        const msg = String(e?.message || e);

        // Cloudflare returns success via error text
        if (msg.includes("Thank you for agreeing")) {
          return Response.json({ ok: true, agreed: true });
        }

        console.error("AGREE FAILED", e);
        return Response.json({ ok: false, error: msg }, { status: 500 });
      }
    }
    
    if (url.pathname === "/capture" && request.method === "POST") {
      const debug = url.searchParams.get("debug") === "1";
      try {
        const form = await request.formData();
        const trunc = (s: string, n = 2000) => (s.length > n ? s.slice(0, n) + "…" : s);

        const goalVal = form.get("goal");
        const imageVal = form.get("image");
        const agentNameVal = form.get("agent_name");

        // 0) Ensure Workers AI binding exists
        if (!env.AI) {
          return Response.json(
            { ok: false, error: "Missing AI binding (env.AI)." },
            { status: 500 }
          );
        }

        // 1) Validate goal first
        if (
          typeof goalVal !== "string" ||
          goalVal.trim().split(/\s+/).length < 2
        ) {
          return Response.json(
            { ok: false, error: "Goal must be at least 2 words." },
            { status: 400 }
          );
        }
        const goal = goalVal.trim();

        // 2) Validate file existence + type
        if (!(imageVal instanceof File)) {
          return Response.json(
            { ok: false, error: "Missing image file." },
            { status: 400 }
          );
        }
        const image = imageVal;

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

        // 3) Build data URL AFTER validation
        const bytes = new Uint8Array(await image.arrayBuffer());
        // Prefer an explicit agent session key from the client (ties /capture to the agent instance name)
        const agentName = typeof agentNameVal === "string" ? agentNameVal.trim() : "";

        let sid = agentName || getCookie(request, "sid");
        const isNewSid = !getCookie(request, "sid") || (agentName && getCookie(request, "sid") !== agentName);
        console.log("[/capture] agent_name:", agentName, "sid:", sid, "cookie_sid:", getCookie(request, "sid"));
        if (!sid) sid = makeSid();
        console.log("[/capture] DO write key:", sid);

        const imageHash = await sha256Base64(bytes);
        const base64 = toBase64(bytes);
        const dataUrl = `data:${image.type};base64,${base64}`;
        // 4) Pick models
        const visionModelId =
          env.VISION_MODEL_ID || "@cf/meta/llama-3.2-11b-vision-instruct";

        // Use a *text* model for structuring (avoid vision model JSON failures)
        const structureModelId = env.STRUCTURE_MODEL_ID || "@cf/meta/llama-3.1-8b-instruct";
        // Helper: convert Workers AI outputs to a string safely
        const toAiText = (out: any) => {
          if (typeof out === "string") return out;
          if (typeof out?.response === "string") return out.response;
          if (typeof out?.result === "string") return out.result;
          if (typeof out?.response?.text === "string") return out.response.text;
          return JSON.stringify(out);
        };

        // --------------------
        // PASS 1: Vision -> TEXT
        // --------------------
        let visionOut: any;
        try {
const visionSystem =
  "You are describing a UI screenshot for a navigation assistant. TEXT ONLY. NO JSON. NO MARKDOWN.\n" +
  "Goal matters: prioritize elements that help achieve the Goal.\n" +
  "Use the exact headings below. Keep each bullet short.\n\n" +
  "SUMMARY: <1 sentence>\n" +
  "PAGE TYPE: <profile/settings/editor/problem/etc>\n" +
  "LAYOUT: <header/sidebar/main/right-rail/modals present?>\n" +
  "GOAL TARGET (MOST IMPORTANT):\n" +
  "- Element: <exact label text if visible>\n" +
  "- Where: <left/right/top/bottom + within which panel/card>\n" +
  "- Looks like: <button/link/icon, color/shape, any icon>\n" +
  "- Nearby: <closest text labels around it>\n" +
  "NAV/TABS:\n" +
  "- <tab item> — <where>\n" +
  "CLICKABLES (TOP 12):\n" +
  "- <label> — <type> — <where> — <what it likely does>\n" +
  "FIELDS (TOP 10):\n" +
  "- <label/placeholder> — <where>\n" +
  "STATUS/ERRORS:\n" +
  "- <anything notable>\n" +
  "TEXT SNIPPETS (TOP 20):\n" +
  "- <important visible text>\n\n" +
  "If the goal target is NOT visible, say: GOAL TARGET NOT VISIBLE.";

          visionOut = await env.AI.run(
            visionModelId as any,
            {
              temperature: 0.2,
              max_tokens: 700,
              messages: [
                { role: "system", content: visionSystem },
                {
                  role: "user",
                  content: [
                    { type: "text", text: `Goal: ${goal}` },
                    { type: "image_url", image_url: { url: dataUrl } }
                  ]
                }
              ]
            } as any
          );
        } catch (e: any) {
          const msg = String(e?.message ?? e);
          console.error("VISION RUN FAILED", e);
          return Response.json(
            {
              ok: false,
              error: msg,
              ...(debug
                ? {
                    debug_stage: "vision",
                    debug_vision_model: visionModelId,
                    debug_goal: goal,
                    debug_image: { type: image.type, size: image.size }
                  }
                : {})
            },
            { status: 502 }
          );
        }

        const visionText = toAiText(visionOut).trim();
        const visionTextClamped = visionText.slice(0, 2000); // keep small so pass 2 doesn't blow up
        // --------------------
        // PASS 2: Text -> JSON
        // --------------------
        let structureOut: any;
        let rawJsonText = "";

        const schemaText =
          "Return ONLY valid JSON (no markdown, no prose) matching EXACTLY this schema:\n" +
          "{\n" +
          '  "screen_summary": string,\n' +
          '  "ui_elements": [{"label": string, "type": string, "hint": string}],\n' +
          '  "steps": [string],\n' +
          '  "confidence": number,\n' +
          '  "need_new_screenshot": boolean,\n' +
          '  "expected_next_screen": string\n' +
          "}\n" +
          "Hard limits (must comply):\n" +
          "- screen_summary <= 160 chars\n" +
          "- ui_elements length between 10 and 14\n" +
            "- each ui_elements.label <= 40 chars\n" +
          "- steps length between 6 and 10\n" +
          "- each step <= 90 chars\n" +
          "- confidence is 0..1\n";

        const runStructure = async (strict: boolean) => {
          const structSystemBase =
            "You are a strict JSON generator. " +
            "Never include <think> tags. Never include markdown. Never include any text outside JSON.";

          const structSystem = strict
            ? structSystemBase +
              " CRITICAL: Keep output short and COMPLETE. " +
              "If you are unsure, use placeholders but still return valid JSON that meets the hard limits."
            : structSystemBase;

          const structUser =
            `${schemaText}\n` +
            `Goal: ${goal}\n\n` +
            `Screenshot description:\n${visionTextClamped}\n`;

          const out = await env.AI.run(
            structureModelId as any,
            {
              temperature: strict ? 0.1 : 0.2,
              max_tokens: 900,
              messages: [
                { role: "system", content: structSystem },
                { role: "user", content: structUser }
              ]
            } as any
          );

          return out;
        };

        try {
          structureOut = await runStructure(false);
          rawJsonText = toAiText(structureOut).trim();
        } catch (e: any) {
          const msg = String(e?.message ?? e);
          console.error("STRUCTURE RUN FAILED", e);
          return Response.json(
            {
              ok: false,
              error: msg,
              ...(debug
                ? {
                    debug_stage: "structure",
                    debug_structure_model: structureModelId,
                    debug_vision_text: visionText.slice(0, 2000)
                  }
                : {})
            },
            { status: 502 }
          );
        }

        // Parse JSON robustly (handles accidental leading/trailing prose).
        // If parsing fails, retry once with stricter constraints.
        let extracted = extractJsonObject(rawJsonText) ?? rawJsonText;

        let candidate: any;
        try {
          candidate = JSON.parse(extracted);
        } catch {
          try {
            structureOut = await runStructure(true);
            rawJsonText = toAiText(structureOut).trim();
            extracted = extractJsonObject(rawJsonText) ?? rawJsonText;
            candidate = JSON.parse(extracted);
          } catch {
            console.error("MODEL JSON PARSE FAILED (truncated):", extracted.slice(0, 1200));
            return Response.json(
              {
                ok: false,
                error: "Model returned invalid JSON.",
                ...(debug
                  ? {
                      debug_stage: "json_parse",
                      debug_structure_model: structureModelId,
                      debug_raw: rawJsonText.slice(0, 3000),
                      debug_extracted: extracted.slice(0, 3000),
                      debug_vision_text: visionText.slice(0, 2000)
                    }
                  : {})
              },
              { status: 502 }
            );
          }
        }

        // Coerce/pad defensively so UI never breaks
        const toStr = (v: any, fallback = "") =>
          typeof v === "string" ? v : v == null ? fallback : String(v);

        const clamp01 = (n: any) => {
          const x = typeof n === "number" ? n : Number(n);
          if (Number.isNaN(x)) return 0.5;
          return Math.max(0, Math.min(1, x));
        };

        const ensureArray = (v: any) => (Array.isArray(v) ? v : []);

        const brain: any = {
          screen_summary: toStr(candidate?.screen_summary, ""),
          ui_elements: ensureArray(candidate?.ui_elements).map((el: any) => ({
            label: toStr(el?.label, "Unknown"),
            type: toStr(el?.type, "unknown"),
            hint: toStr(el?.hint, "")
          })),
          steps: ensureArray(candidate?.steps).map((s: any) => toStr(s, "")),
          confidence: clamp01(candidate?.confidence),
          need_new_screenshot: Boolean(candidate?.need_new_screenshot),
          expected_next_screen: toStr(candidate?.expected_next_screen, "Unknown")
        };

        if (!brain.screen_summary) brain.screen_summary = "(No summary provided)";

        while (brain.ui_elements.length < 10) {
          brain.ui_elements.push({ label: "Unknown", type: "unknown", hint: "" });
        }

        brain.steps = brain.steps.filter((s: string) => s.trim().length > 0);
        while (brain.steps.length < 6) {
          brain.steps.push("(Step missing from model; user may need a clearer screenshot.)");
        }
                // Persist richer capture artifacts for later chat turns (bounded)
        brain.vision_text = visionText.slice(0, 4000);
        brain.raw_model_json = rawJsonText.slice(0, 6000);

        // 6) Save to Durable Object
        let do_saved = false;
        let do_error: string | null = null;
        try {
          const id = env.SESSION_MEMORY.idFromName(sid);
          const stub = env.SESSION_MEMORY.get(id);

          await stub.fetch("https://do/session/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              goal,
              image_hash: imageHash,
              brain
            })
          });

          do_saved = true;
        } catch (e: any) {
          do_error = String(e?.message ?? e);
          console.error("DO SAVE FAILED", e);
          // Do not fail the request; we still want to show the model result in the UI.
        }

        // Keep the default /capture response small so the UI doesn't dump huge JSON into chat.
        // Full details remain available when debug=1.
        const minimal = {
          ok: true,
          sid,
          do_saved,
          message:
            `Screenshot received. ${brain.screen_summary} ` +
            `Confidence: ${brain.confidence}. ` +
            `Use GET /session to fetch full structured data.`,
          brain_summary: brain.screen_summary,
          confidence: brain.confidence,
          need_new_screenshot: brain.need_new_screenshot,
          expected_next_screen: brain.expected_next_screen
        };

        const full = {
          ok: true,
          sid,
          received: {
            filename: image.name,
            type: image.type,
            size: image.size,
            goal
          },
          brain,
          do_saved,
          ...(debug
            ? {
                do_error,
                debug: {
                  vision_model: visionModelId,
                  structure_model: structureModelId,
                  goal,
                  image: { name: image.name, type: image.type, size: image.size },
                  vision_text: trunc(visionText, 2000),
                  raw_json_text: trunc(rawJsonText, 3000),
                  extracted_json: trunc(extracted, 3000),
                  // Raw model outputs (can be large)
                  vision_out: visionOut,
                  structure_out: structureOut
                }
              }
            : {})
        };

        const resp = Response.json(debug ? full : minimal);

        if (isNewSid || (typeof agentNameVal === "string" && agentNameVal.trim().length > 0)) {
          resp.headers.set(
            "Set-Cookie",
            `sid=${encodeURIComponent(sid)}; Path=/; HttpOnly; SameSite=Lax`
          );
        }

        return resp;
      } catch (e: any) {
        console.error("CAPTURE FAILED", e);
        const msg = String(e?.message ?? e);
        return Response.json(
          {
            ok: false,
            error: msg,
            ...(debug ? { debug_stack: String(e?.stack ?? "") } : {})
          },
          { status: 502 }
        );
      }
    }

    if (url.pathname === "/session" && request.method === "GET") {
      const sidFromQuery = url.searchParams.get("sid")?.trim();
      const sidFromHeader = request.headers.get("x-sid")?.trim();
      const sidFromCookie = getCookie(request, "sid");
      const sid = sidFromQuery || sidFromHeader || sidFromCookie;

      if (!sid) {
        return Response.json(
          { ok: false, error: "No sid provided (query/header/cookie)." },
          { status: 400 }
        );
      }

      const id = env.SESSION_MEMORY.idFromName(sid);
      const stub = env.SESSION_MEMORY.get(id);
      return stub.fetch("https://do/session", { method: "GET" });
    }

    if (url.pathname === "/session/reset" && request.method === "POST") {
      const sidFromQuery = url.searchParams.get("sid")?.trim();
      const sidFromHeader = request.headers.get("x-sid")?.trim();
      const sidFromCookie = getCookie(request, "sid");
      const sid = sidFromQuery || sidFromHeader || sidFromCookie;

      if (!sid) {
        return Response.json(
          { ok: false, error: "No sid provided (query/header/cookie)." },
          { status: 400 }
        );
      }

      const id = env.SESSION_MEMORY.idFromName(sid);
      const stub = env.SESSION_MEMORY.get(id);
      return stub.fetch("https://do/session/reset", { method: "POST" });
    }
        if (url.pathname === "/session/memo" && request.method === "POST") {
      const sidFromQuery = url.searchParams.get("sid")?.trim();
      const sidFromHeader = request.headers.get("x-sid")?.trim();
      const sidFromCookie = getCookie(request, "sid");
      const sid = sidFromQuery || sidFromHeader || sidFromCookie;

      if (!sid) {
        return Response.json(
          { ok: false, error: "No sid provided (query/header/cookie)." },
          { status: 400 }
        );
      }

      let body: any = null;
      try {
        body = await request.json();
      } catch {
        body = null;
      }

      const memo = typeof body?.memo === "string" ? body.memo : "";
      const mode = body?.mode === "append" ? "append" : "replace";

      const id = env.SESSION_MEMORY.idFromName(sid);
      const stub = env.SESSION_MEMORY.get(id);
      return stub.fetch("https://do/session/memo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memo, mode })
      });
    }
    return (
      // Route the request to our agent or return 404 if not found
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;

function toBase64(bytes: Uint8Array) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function getCookie(req: Request, name: string): string | null {
  const cookie = req.headers.get("Cookie") || "";
  const m = cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return m ? decodeURIComponent(m[1]) : null;
}

function makeSid(): string {
  return crypto.randomUUID();
}

async function sha256Base64(bytes: Uint8Array): Promise<string> {
  // Force a real ArrayBuffer (never SharedArrayBuffer) by copying
  const ab = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(ab).set(bytes);

  const digest = await crypto.subtle.digest("SHA-256", ab);

  const arr = new Uint8Array(digest);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < arr.length; i += chunkSize) {
    binary += String.fromCharCode(...arr.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function extractJsonObject(text: string): string | null {
  // Finds the first top-level JSON object in a string (handles leading/trailing prose).
  const start = text.indexOf("{");
  if (start === -1) return null;

  let depth = 0;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (ch === "{") depth++;
    else if (ch === "}") depth--;

    if (depth === 0) {
      return text.slice(start, i + 1);
    }
  }
  return null;
}