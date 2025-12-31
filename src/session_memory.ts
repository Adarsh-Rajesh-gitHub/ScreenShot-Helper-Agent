import { DurableObject } from "cloudflare:workers";

export type Brain = {
  screen_summary: string;
  ui_elements: Array<{ label: string; type: string; hint: string }>;
  steps: string[];
  confidence: number;
  need_new_screenshot: boolean;
  expected_next_screen: string;
  // Optional: richer capture artifacts (not required by the UI)
  vision_text?: string; // raw text description from the vision pass (if stored)
  raw_model_json?: string; // raw JSON (or near-JSON) emitted by the structuring pass (if stored)
};

export type SessionState = {
  last_image_hash: string | null;
  last_result_json: Brain | null;
  expected_next_screen: string | null;
  step_index: number;
  history: Array<{
    ts: number;
    goal: string;
    image_hash: string;
    brain: Brain;
  }>;
  // Agent-owned working memory (never rendered directly to the user)
  active_goal: string | null;
  agent_memo: string | null;
  memo_ts: number | null;
};

const DEFAULT_STATE: SessionState = {
  last_image_hash: null,
  last_result_json: null,
  expected_next_screen: null,
  step_index: 0,
  history: [],
  active_goal: null,
  agent_memo: null,
  memo_ts: null
};

export class SessionMemoryDO extends DurableObject {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/session") {
      const s = await this.readState();
      return Response.json({ ok: true, session: s });
    }

    // POST /session/update -> store capture result
    if (request.method === "POST" && url.pathname === "/session/update") {
      const body = (await request.json()) as {
        goal: string;
        image_hash: string;
        brain: Brain;
      };

      const s = await this.readState();

      const next: SessionState = {
        ...s,
        last_image_hash: body.image_hash,
        last_result_json: body.brain,
        expected_next_screen: body.brain.expected_next_screen ?? null,
        step_index: 0,
        active_goal: body.goal,
        agent_memo: null,
        memo_ts: null,
        history: [
          ...s.history,
          {
            ts: Date.now(),
            goal: body.goal,
            image_hash: body.image_hash,
            brain: body.brain
          }
        ].slice(-10) // cap to last 10
      };

      await this.ctx.storage.put("state", next);
      return Response.json({ ok: true });
    }

    // POST /session/reset -> clears everything atomically
    if (request.method === "POST" && url.pathname === "/session/reset") {
      await this.ctx.storage.put("state", DEFAULT_STATE);
      return Response.json({ ok: true });
    }

    // POST /session/memo -> set or append agent memo (hidden working memory)
    if (request.method === "POST" && url.pathname === "/session/memo") {
      const body = (await request.json()) as {
        memo: string;
        mode?: "replace" | "append";
      };

      const s = await this.readState();

      const incoming = typeof body?.memo === "string" ? body.memo : "";
      const mode = body?.mode === "append" ? "append" : "replace";

      const merged =
        mode === "append"
          ? `${s.agent_memo ? s.agent_memo + "\n" : ""}${incoming}`
          : incoming;

      // Keep DO storage bounded
      const MAX_MEMO_CHARS = 8000;
      const clamped =
        merged.length > MAX_MEMO_CHARS ? merged.slice(0, MAX_MEMO_CHARS) + "…" : merged;

      const next: SessionState = {
        ...s,
        agent_memo: clamped.trim().length ? clamped : null,
        memo_ts: Date.now()
      };

      await this.ctx.storage.put("state", next);
      return Response.json({ ok: true });
    }

    // POST /session/step -> update step_index (optional utility)
    if (request.method === "POST" && url.pathname === "/session/step") {
      const body = (await request.json()) as { step_index: number };
      const s = await this.readState();
      const next = { ...s, step_index: Math.max(0, body.step_index | 0) };
      await this.ctx.storage.put("state", next);
      return Response.json({ ok: true });
    }

    return new Response("Not found", { status: 404 });
  }

  private async readState(): Promise<SessionState> {
    const stored = await this.ctx.storage.get<SessionState>("state");
    // Merge to ensure newly-added fields always exist even for older persisted states.
    return { ...DEFAULT_STATE, ...(stored ?? {}) };
  }
}