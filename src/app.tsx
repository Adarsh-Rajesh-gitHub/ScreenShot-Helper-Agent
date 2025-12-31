/** biome-ignore-all lint/correctness/useUniqueElementIds: it's alright */
import { useEffect, useState, useRef, useCallback, use } from "react";
import { useAgent } from "agents/react";
import { isToolUIPart } from "ai";
import { useAgentChat } from "agents/ai-react";
import type { UIMessage } from "@ai-sdk/react";
import type { tools } from "./tools";

// Component imports
import { Button } from "@/components/button/Button";
import { Card } from "@/components/card/Card";
import { Avatar } from "@/components/avatar/Avatar";
import { Toggle } from "@/components/toggle/Toggle";
import { Textarea } from "@/components/textarea/Textarea";
import { MemoizedMarkdown } from "@/components/memoized-markdown";
import { ToolInvocationCard } from "@/components/tool-invocation-card/ToolInvocationCard";

// Icon imports
import {
  Bug,
  Moon,
  Robot,
  Sun,
  Trash,
  PaperPlaneTilt,
  Stop
} from "@phosphor-icons/react";

// List of tools that require human confirmation
// NOTE: this should match tools that do NOT have execute functions in tools.ts.
// This project intentionally exposes no confirmation-only tools.
const toolsRequiringConfirmation: (keyof typeof tools)[] = [];

export default function Chat() {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    // Check localStorage first, default to dark if not found
    const savedTheme = localStorage.getItem("theme");
    return (savedTheme as "dark" | "light") || "dark";
  });
  const [showDebug, setShowDebug] = useState(false);
  const [textareaHeight, setTextareaHeight] = useState("auto");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Stable agent/session name for this browser. This MUST match the key used by the backend
  // (server.ts uses `this.name` as the session key, and /capture expects `agent_name`).
  const [agentName] = useState(() => {
    const key = "agents-starter:agentName";
    const existing = localStorage.getItem(key);
    if (existing) return existing;

    const created =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `sid_${Date.now()}_${Math.random().toString(16).slice(2)}`;

    localStorage.setItem(key, created);
    return created;
  });


  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    // Apply theme class on mount and when theme changes
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }

    // Save theme preference to localStorage
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Scroll to bottom on mount
  useEffect(() => {
    scrollToBottom();
  }, [scrollToBottom]);

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
  };

  const agent = useAgent({
    agent: "chat",
    // NOTE: types for useAgent may not include `name`, but the runtime supports named DO instances.
    // We cast to `any` to ensure the correct instance is used.
    name: agentName
  } as any);

  const [agentInput, setAgentInput] = useState("");
  // B1 Capture state (no AI call yet)
  const [goal, setGoal] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [captureResp, setCaptureResp] = useState<any>(null);
  const [captureErr, setCaptureErr] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [sessionResp, setSessionResp] = useState<any>(null);
  const [sessionErr, setSessionErr] = useState<string | null>(null);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [showCaptureRaw, setShowCaptureRaw] = useState(false);

  const {
    messages: agentMessages,
    addToolResult,
    clearHistory,
    status,
    sendMessage,
    stop
  } = useAgentChat<unknown, UIMessage<{ createdAt: string }>>({
    agent
  });

  async function fetchSessionForContext() {
    try {
      const r = await fetch(`/session?sid=${encodeURIComponent(agentName)}`, {
        method: "GET",
        credentials: "include",
        headers: {
          "x-sid": agentName
        }
      });
      if (!r.ok) return null;
      return await r.json().catch(() => null);
    } catch {
      return null;
    }
  }

  function buildHiddenScreenshotContext(sessionPayload: any) {
    const s = sessionPayload?.session ?? sessionPayload;
    const brain = s?.last_result_json;
    if (!brain) return null;

    const lastGoal =
      Array.isArray(s?.history) && s.history.length
        ? s.history[s.history.length - 1]?.goal
        : goal.trim();

    const elements = Array.isArray(brain.ui_elements) ? brain.ui_elements : [];
    const steps = Array.isArray(brain.steps) ? brain.steps : [];

    const elementsBlock =
      elements
        .slice(0, 12)
        .map((e: any) => {
          const label = e?.label ?? "?";
          const type = e?.type ?? "?";
          const hint = e?.hint ? `: ${e.hint}` : "";
          return `- ${label} [${type}]${hint}`;
        })
        .join("\n") || "(none)";

    const stepsBlock =
      steps
        .slice(0, 8)
        .map((x: string, i: number) => `${i + 1}. ${x}`)
        .join("\n") || "(none)";

    return (
      `[[SCREENSHOT_CONTEXT]]\n` +
      `Goal: ${lastGoal || "(unknown)"}\n` +
      `Screen summary: ${brain.screen_summary || "(none)"}\n` +
      `Confidence: ${brain.confidence ?? "(n/a)"}\n` +
      `Need new screenshot: ${brain.need_new_screenshot ? "true" : "false"}\n` +
      `Expected next screen: ${brain.expected_next_screen || "(unknown)"}\n` +
      `UI elements:\n${elementsBlock}\n` +
      `Steps:\n${stepsBlock}\n` +
      `[[/SCREENSHOT_CONTEXT]]\n`
    );
  }

  async function loadLatestSession() {
    setSessionErr(null);
    setSessionResp(null);
    setSessionLoading(true);
    try {
      const r = await fetch(`/session?sid=${encodeURIComponent(agentName)}`, {
        method: "GET",
        credentials: "include",
        headers: {
          "x-sid": agentName
        }
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((data as any)?.error || `Session fetch failed (${r.status})`);
      setSessionResp(data);
    } catch (err: any) {
      setSessionErr(err?.message || "Failed to load session.");
    } finally {
      setSessionLoading(false);
    }
  }

  async function resetSession() {
    setSessionErr(null);
    setSessionResp(null);
    setSessionLoading(true);
    try {
      const r = await fetch(`/session/reset?sid=${encodeURIComponent(agentName)}`, {
        method: "POST",
        credentials: "include",
        headers: { "x-sid": agentName }
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok)
        throw new Error((data as any)?.error || `Reset failed (${r.status})`);
      // Reload so the UI can self-verify the cleared state.
      await loadLatestSession();
    } catch (err: any) {
      setSessionErr(err?.message || "Failed to reset session.");
    } finally {
      setSessionLoading(false);
    }
  }

  function buildScreenshotContext(goal: string, capture: any) {
    const brain = capture?.brain;

    // Prefer backend-provided compact summary if present
    if (!brain) {
      const fallbackSummary = capture?.brain_summary;
      return [
        "Screenshot analysis context (use this instead of the raw image):",
        `Goal: ${goal}`,
        `Summary: ${fallbackSummary ?? "(no analysis returned)"}`
      ].join("\n");
    }

    const elements = Array.isArray(brain.ui_elements) ? brain.ui_elements : [];
    const steps = Array.isArray(brain.steps) ? brain.steps : [];

    const elementsLine =
      elements
        .slice(0, 10)
        .map((e: any) => `${e?.label ?? "?"} (${e?.type ?? "?"})`)
        .join(", ") || "(none)";

    const stepsBlock =
      steps
        .slice(0, 8)
        .map((s: string, i: number) => `${i + 1}. ${s}`)
        .join("\n") || "(none)";

    return [
      "Screenshot analysis context (use this instead of the raw image):",
      `Goal: ${goal}`,
      `Screen summary: ${brain?.screen_summary ?? "(none)"}`,
      `UI elements: ${elementsLine}`,
      `Steps:\n${stepsBlock}`,
      `Confidence: ${brain?.confidence ?? "(n/a)"}`
    ].join("\n");
  }

  async function handleCaptureSubmit(e: React.FormEvent) {
    e.preventDefault();
    setCaptureErr(null);
    setCaptureResp(null);
    setSessionErr(null);
    setSessionResp(null);
    setShowCaptureRaw(false);

    if (!imageFile) {
      setCaptureErr("Missing image.");
      return;
    }

    const trimmed = goal.trim();
    if (!trimmed) {
      setCaptureErr("Missing goal.");
      return;
    }

    if (trimmed.split(/\s+/).length < 2) {
      setCaptureErr("Goal must be at least 2 words.");
      return;
    }

    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("goal", trimmed);
      fd.append("image", imageFile);
      // Tie capture storage to the same durable session key used by the chat agent.
      fd.append("agent_name", agentName);

      const r = await fetch("/capture", {
        method: "POST",
        body: fd,
        credentials: "include",
        headers: {
          "x-sid": agentName
        }
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((data as any)?.error || `Upload failed (${r.status})`);
      setCaptureResp(data);

      // Authoritative session key is the agentName (must match the DO name).
      // Server may also return a sid, but we always use agentName to avoid mismatches.
      // UX: don't auto-send into chat. Prime the chat input so the next Send uses the stored screenshot context.
      setAgentInput(trimmed);

      // Try to load the server-stored session (if the backend exposes it)
      loadLatestSession();
    } catch (err: any) {
      setCaptureErr(err?.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }
  const handleAgentInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setAgentInput(e.target.value);
  };

const handleAgentSubmit = async (
  e: React.FormEvent,
  extraData: Record<string, unknown> = {}
) => {
  e.preventDefault();
  if (!agentInput.trim()) return;

  const message = agentInput;
  setAgentInput("");

  let hiddenContext: string | null = null;
  if (captureResp) {
    const sessionPayload = await fetchSessionForContext();
    hiddenContext = sessionPayload ? buildHiddenScreenshotContext(sessionPayload) : null;
  }

  const parts: any[] = [];
  if (hiddenContext) parts.push({ type: "text", text: hiddenContext }); // hidden by your renderer
  parts.push({ type: "text", text: message });

  try {
    await sendMessage(
      { role: "user", parts } as any,
      {
        body: {
          ...extraData,
          agent_name: agentName,
          sid: agentName,
          captureSid: agentName,
          screenshot_context: Boolean(captureResp)
        }
      }
    );
  } catch (err) {
    console.error("sendMessage failed:", err);
    alert("Send failed. Check Console + Network for the first error.");
  }
};

  // Scroll to bottom when messages change
  useEffect(() => {
    agentMessages.length > 0 && scrollToBottom();
  }, [agentMessages, scrollToBottom]);

  const pendingToolCallConfirmation = agentMessages.some((m: UIMessage) =>
    m.parts?.some(
      (part) =>
        isToolUIPart(part) &&
        part.state === "input-available" &&
        // Manual check inside the component
        toolsRequiringConfirmation.includes(
          part.type.replace("tool-", "") as keyof typeof tools
        )
    )
  );

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="h-[100vh] w-full p-4 flex justify-center items-center bg-fixed overflow-hidden">
      <HasOpenAIKey />
      <div className="h-[calc(100vh-2rem)] w-full mx-auto max-w-lg flex flex-col shadow-xl rounded-md overflow-hidden relative border border-neutral-300 dark:border-neutral-800">
        <div className="px-4 py-3 border-b border-neutral-300 dark:border-neutral-800 flex items-center gap-3 sticky top-0 z-10">
          <div className="flex items-center justify-center h-8 w-8">
            <svg
              width="28px"
              height="28px"
              className="text-[#F48120]"
              data-icon="agents"
            >
              <title>Cloudflare Agents</title>
              <symbol id="ai:local:agents" viewBox="0 0 80 79">
                <path
                  fill="currentColor"
                  d="M69.3 39.7c-3.1 0-5.8 2.1-6.7 5H48.3V34h4.6l4.5-2.5c1.1.8 2.5 1.2 3.9 1.2 3.8 0 7-3.1 7-7s-3.1-7-7-7-7 3.1-7 7c0 .9.2 1.8.5 2.6L51.9 30h-3.5V18.8h-.1c-1.3-1-2.9-1.6-4.5-1.9h-.2c-1.9-.3-3.9-.1-5.8.6-.4.1-.8.3-1.2.5h-.1c-.1.1-.2.1-.3.2-1.7 1-3 2.4-4 4 0 .1-.1.2-.1.2l-.3.6c0 .1-.1.1-.1.2v.1h-.6c-2.9 0-5.7 1.2-7.7 3.2-2.1 2-3.2 4.8-3.2 7.7 0 .7.1 1.4.2 2.1-1.3.9-2.4 2.1-3.2 3.5s-1.2 2.9-1.4 4.5c-.1 1.6.1 3.2.7 4.7s1.5 2.9 2.6 4c-.8 1.8-1.2 3.7-1.1 5.6 0 1.9.5 3.8 1.4 5.6s2.1 3.2 3.6 4.4c1.3 1 2.7 1.7 4.3 2.2v-.1q2.25.75 4.8.6h.1c0 .1.1.1.1.1.9 1.7 2.3 3 4 4 .1.1.2.1.3.2h.1c.4.2.8.4 1.2.5 1.4.6 3 .8 4.5.7.4 0 .8-.1 1.3-.1h.1c1.6-.3 3.1-.9 4.5-1.9V62.9h3.5l3.1 1.7c-.3.8-.5 1.7-.5 2.6 0 3.8 3.1 7 7 7s7-3.1 7-7-3.1-7-7-7c-1.5 0-2.8.5-3.9 1.2l-4.6-2.5h-4.6V48.7h14.3c.9 2.9 3.5 5 6.7 5 3.8 0 7-3.1 7-7s-3.1-7-7-7m-7.9-16.9c1.6 0 3 1.3 3 3s-1.3 3-3 3-3-1.3-3-3 1.4-3 3-3m0 41.4c1.6 0 3 1.3 3 3s-1.3 3-3 3-3-1.3-3-3 1.4-3 3-3M44.3 72c-.4.2-.7.3-1.1.3-.2 0-.4.1-.5.1h-.2c-.9.1-1.7 0-2.6-.3-1-.3-1.9-.9-2.7-1.7-.7-.8-1.3-1.7-1.6-2.7l-.3-1.5v-.7q0-.75.3-1.5c.1-.2.1-.4.2-.7s.3-.6.5-.9c0-.1.1-.1.1-.2.1-.1.1-.2.2-.3s.1-.2.2-.3c0 0 0-.1.1-.1l.6-.6-2.7-3.5c-1.3 1.1-2.3 2.4-2.9 3.9-.2.4-.4.9-.5 1.3v.1c-.1.2-.1.4-.1.6-.3 1.1-.4 2.3-.3 3.4-.3 0-.7 0-1-.1-2.2-.4-4.2-1.5-5.5-3.2-1.4-1.7-2-3.9-1.8-6.1q.15-1.2.6-2.4l.3-.6c.1-.2.2-.4.3-.5 0 0 0-.1.1-.1.4-.7.9-1.3 1.5-1.9 1.6-1.5 3.8-2.3 6-2.3q1.05 0 2.1.3v-4.5c-.7-.1-1.4-.2-2.1-.2-1.8 0-3.5.4-5.2 1.1-.7.3-1.3.6-1.9 1s-1.1.8-1.7 1.3c-.3.2-.5.5-.8.8-.6-.8-1-1.6-1.3-2.6-.2-1-.2-2 0-2.9.2-1 .6-1.9 1.3-2.6.6-.8 1.4-1.4 2.3-1.8l1.8-.9-.7-1.9c-.4-1-.5-2.1-.4-3.1s.5-2.1 1.1-2.9q.9-1.35 2.4-2.1c.9-.5 2-.8 3-.7.5 0 1 .1 1.5.2 1 .2 1.8.7 2.6 1.3s1.4 1.4 1.8 2.3l4.1-1.5c-.9-2-2.3-3.7-4.2-4.9q-.6-.3-.9-.6c.4-.7 1-1.4 1.6-1.9.8-.7 1.8-1.1 2.9-1.3.9-.2 1.7-.1 2.6 0 .4.1.7.2 1.1.3V72zm25-22.3c-1.6 0-3-1.3-3-3 0-1.6 1.3-3 3-3s3 1.3 3 3c0 1.6-1.3 3-3 3"
                />
              </symbol>
              <use href="#ai:local:agents" />
            </svg>
          </div>

          <div className="flex-1">
            <h2 className="font-semibold text-base">AI Chat Agent</h2>
          </div>

          <div className="flex items-center gap-2 mr-2">
            <Bug size={16} />
            <Toggle
              toggled={showDebug}
              aria-label="Toggle debug mode"
              onClick={() => setShowDebug((prev) => !prev)}
            />
          </div>

          <Button
            variant="ghost"
            size="md"
            shape="square"
            className="rounded-full h-9 w-9"
            onClick={toggleTheme}
          >
            {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
          </Button>

          <Button
            variant="ghost"
            size="md"
            shape="square"
            className="rounded-full h-9 w-9"
            onClick={clearHistory}
          >
            <Trash size={20} />
          </Button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 pb-24 max-h-[calc(100vh-10rem)]">
          {agentMessages.length === 0 && (
            <div className="h-full flex items-center justify-center">
              <Card className="p-6 max-w-md mx-auto bg-neutral-100 dark:bg-neutral-900">
                <div className="text-center space-y-4">
                  <div className="bg-[#F48120]/10 text-[#F48120] rounded-full p-3 inline-flex">
                    <Robot size={24} />
                  </div>
                  <h3 className="font-semibold text-lg">Welcome to AI Chat</h3>
                  <p className="text-muted-foreground text-sm">
                    Upload a screenshot + goal, then ask for exact click-by-click steps.
                  </p>
                  <ul className="text-sm text-left space-y-2">
                    <li className="flex items-center gap-2">
                      <span className="text-[#F48120]">•</span>
                      <span>"I want to edit my profile" (after uploading the page screenshot)</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-[#F48120]">•</span>
                      <span>"Which button should I click next?"</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-[#F48120]">•</span>
                      <span>"What changed on this screen?" (upload the next screenshot)</span>
                    </li>
                  </ul>
                </div>
              </Card>
            </div>
          )}

          {agentMessages.map((m, index) => {
            const isUser = m.role === "user";
            const showAvatar =
              index === 0 || agentMessages[index - 1]?.role !== m.role;

            return (
              <div key={m.id}>
                {showDebug && (
                  <pre className="text-xs text-muted-foreground overflow-scroll">
                    {JSON.stringify(m, null, 2)}
                  </pre>
                )}
                <div
                  className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`flex gap-2 max-w-[85%] ${
                      isUser ? "flex-row-reverse" : "flex-row"
                    }`}
                  >
                    {showAvatar && !isUser ? (
                      <Avatar username={"AI"} className="flex-shrink-0" />
                    ) : (
                      !isUser && <div className="w-8" />
                    )}

                    <div>
                      <div>
                        {m.parts?.map((part, i) => {
                          if (part.type === "text") {
                            const raw = part.text ?? "";

                            // Hide the injected screenshot context entirely
                            if (raw.startsWith("[[SCREENSHOT_CONTEXT]]")) return null;

                            // Hide agent memo JSON (strip it from display)
                            const memoMarker = "[[AGENT_MEMO]]";
                            const memoIdx = raw.indexOf(memoMarker);

                            const visible = memoIdx === -1 ? raw : raw.slice(0, memoIdx).trimEnd();
                            if (!visible) return null;

                            return (
                              // biome-ignore lint/suspicious/noArrayIndexKey: immutable index
                              <div key={i}>
                                <Card
                                  className={`p-3 rounded-md bg-neutral-100 dark:bg-neutral-900 ${
                                    isUser ? "rounded-br-none" : "rounded-bl-none border-assistant-border"
                                  } ${
                                    visible.startsWith("scheduled message") ? "border-accent/50" : ""
                                  } relative`}
                                >
                                  {visible.startsWith("scheduled message") && (
                                    <span className="absolute -top-3 -left-2 text-base">🕒</span>
                                  )}
                                  <MemoizedMarkdown
                                    id={`${m.id}-${i}`}
                                    content={visible.replace(/^scheduled message: /, "")}
                                  />
                                </Card>
                                <p
                                  className={`text-xs text-muted-foreground mt-1 ${
                                    isUser ? "text-right" : "text-left"
                                  }`}
                                >
                                  {formatTime(
                                    m.metadata?.createdAt
                                      ? new Date(m.metadata.createdAt)
                                      : new Date()
                                  )}
                                </p>
                              </div>
                            );
                          }

                          if (isToolUIPart(part) && m.role === "assistant") {
                            const toolCallId = part.toolCallId;
                            const toolName = part.type.replace("tool-", "");
                            const needsConfirmation =
                              toolsRequiringConfirmation.includes(
                                toolName as keyof typeof tools
                              );

                            return (
                              <ToolInvocationCard
                                // biome-ignore lint/suspicious/noArrayIndexKey: using index is safe here as the array is static
                                key={`${toolCallId}-${i}`}
                                toolUIPart={part}
                                toolCallId={toolCallId}
                                needsConfirmation={needsConfirmation}
                                onSubmit={({ toolCallId, result }) => {
                                  addToolResult({
                                    tool: part.type.replace("tool-", ""),
                                    toolCallId,
                                    output: result
                                  });
                                }}
                                addToolResult={(toolCallId, result) => {
                                  addToolResult({
                                    tool: part.type.replace("tool-", ""),
                                    toolCallId,
                                    output: result
                                  });
                                }}
                              />
                            );
                          }
                          return null;
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>

        {/* Capture (B1): upload screenshot + goal, no AI call yet */}
        <div className="px-4 py-3 border-t border-neutral-300 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900">
          <form onSubmit={handleCaptureSubmit} className="flex flex-col gap-2">
            <div className="flex flex-col gap-1">
              <label className="text-sm text-neutral-600 dark:text-neutral-400">
                Goal (one sentence)
              </label>
              <input
                name="goal"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="e.g., Reset my password on this site."
                className="w-full rounded-md border border-neutral-200 dark:border-neutral-800 bg-transparent px-3 py-2"
                required
              />
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-sm text-neutral-600 dark:text-neutral-400">
                Screenshot (PNG/JPG)
              </label>
                          <div
              onDragOver={(e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = "copy";
              }}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files?.[0];
                if (!f) return;
                if (!["image/png", "image/jpeg"].includes(f.type)) {
                  setCaptureErr("Only PNG/JPG allowed.");
                  return;
                }
                setCaptureErr(null);
                setImageFile(f);
              }}
              className="rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 p-3 text-sm text-neutral-600 dark:text-neutral-400"
            >
              Drag & drop a PNG/JPG here
              {imageFile ? ` (selected: ${imageFile.name})` : ""}
            </div>
              <input
                type="file"
                name="image"
                accept="image/png,image/jpeg"
                onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
                required
              />
            </div>

            <div className="flex items-center gap-2">
              <button
                type="submit"
                disabled={uploading}
                className="rounded-md border border-neutral-200 dark:border-neutral-800 px-3 py-2"
              >
                {uploading ? "Uploading..." : "Send screenshot"}
              </button>

              {captureErr && (
                <span className="text-sm text-red-600">{captureErr}</span>
              )}
              {captureResp && (
                <span className="text-sm text-green-600">Received.</span>
              )}
            </div>

            {captureResp && (
              <div className="mt-2">
                <div className="text-xs text-neutral-600 dark:text-neutral-400">
                  {captureResp.brain_summary
                    ? `Summary: ${captureResp.brain_summary}`
                    : "Capture succeeded."}
                </div>

                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={loadLatestSession}
                    disabled={sessionLoading}
                    className="rounded-md border border-neutral-200 dark:border-neutral-800 px-2 py-1 text-xs"
                  >
                    {sessionLoading ? "Loading..." : "Load analysis"}
                  </button>
                  <button
                    type="button"
                    onClick={resetSession}
                    disabled={sessionLoading}
                    className="rounded-md border border-neutral-200 dark:border-neutral-800 px-2 py-1 text-xs"
                  >
                    Reset analysis
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowCaptureRaw((v) => !v)}
                    disabled={!showDebug}
                    title={showDebug ? "Toggle raw capture response" : "Enable Debug to view raw JSON"}
                    className={`rounded-md border border-neutral-200 dark:border-neutral-800 px-2 py-1 text-xs ${
                      showDebug ? "" : "opacity-50 cursor-not-allowed"
                    }`}
                  >
                    {showCaptureRaw ? "Hide raw" : "Show raw"}
                  </button>
                </div>

                {sessionErr && (
                  <div className="mt-2 text-xs text-red-600">{sessionErr}</div>
                )}

                {sessionResp && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs text-neutral-600 dark:text-neutral-400">
                      Analysis details
                    </summary>
                    <pre className="mt-2 text-xs overflow-auto rounded-md border border-neutral-200 dark:border-neutral-800 p-2">
                      {JSON.stringify(
                        (sessionResp as any)?.session ?? sessionResp,
                        null,
                        2
                      )}
                    </pre>
                  </details>
                )}

                {showDebug && showCaptureRaw && (
                  <details className="mt-2" open>
                    <summary className="cursor-pointer text-xs text-neutral-600 dark:text-neutral-400">
                      Raw capture response
                    </summary>
                    <pre className="mt-2 text-xs overflow-auto rounded-md border border-neutral-200 dark:border-neutral-800 p-2">
                      {JSON.stringify(captureResp, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </form>
        </div>
        {/* Input Area */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleAgentSubmit(e, {
              annotations: {
                hello: "world"
              }
            });
            setTextareaHeight("auto"); // Reset height after submission
          }}
  className="p-3 bg-neutral-50 border-t border-neutral-300 dark:border-neutral-800 dark:bg-neutral-900"        >
          <div className="flex items-center gap-2">
            <div className="flex-1 relative">
              <Textarea
                disabled={pendingToolCallConfirmation}
                placeholder={
                  pendingToolCallConfirmation
                    ? "Please respond to the tool confirmation above..."
                    : "Send a message..."
                }
                className="flex w-full border border-neutral-200 dark:border-neutral-700 px-3 py-2  ring-offset-background placeholder:text-neutral-500 dark:placeholder:text-neutral-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300 dark:focus-visible:ring-neutral-700 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-neutral-900 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm min-h-[24px] max-h-[calc(75dvh)] overflow-hidden resize-none rounded-2xl !text-base pb-10 dark:bg-neutral-900"
                value={agentInput}
                onChange={(e) => {
                  handleAgentInputChange(e);
                  // Auto-resize the textarea
                  e.target.style.height = "auto";
                  e.target.style.height = `${e.target.scrollHeight}px`;
                  setTextareaHeight(`${e.target.scrollHeight}px`);
                }}
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    !e.shiftKey &&
                    !e.nativeEvent.isComposing
                  ) {
                    e.preventDefault();
                    handleAgentSubmit(e as unknown as React.FormEvent);
                    setTextareaHeight("auto"); // Reset height on Enter submission
                  }
                }}
                rows={2}
                style={{ height: textareaHeight }}
              />
              <div className="absolute bottom-0 right-0 p-2 w-fit flex flex-row justify-end">
                {status === "submitted" || status === "streaming" ? (
                  <button
                    type="button"
                    onClick={stop}
                    className="inline-flex items-center cursor-pointer justify-center gap-2 whitespace-nowrap text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 bg-primary text-primary-foreground hover:bg-primary/90 rounded-full p-1.5 h-fit border border-neutral-200 dark:border-neutral-800"
                    aria-label="Stop generation"
                  >
                    <Stop size={16} />
                  </button>
                ) : (
                  <button
                    type="submit"
                    className="inline-flex items-center cursor-pointer justify-center gap-2 whitespace-nowrap text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 bg-primary text-primary-foreground hover:bg-primary/90 rounded-full p-1.5 h-fit border border-neutral-200 dark:border-neutral-800"
                    disabled={pendingToolCallConfirmation || !agentInput.trim()}
                    aria-label="Send message"
                  >
                    <PaperPlaneTilt size={16} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

const hasOpenAiKeyPromise = (async () => {
  // Prefer Workers AI binding check. Fall back to older OpenAI-key check
  // so the UI still works if server.ts hasn't been updated yet.
  const tryFetch = async (path: string) => {
    const res = await fetch(path);
    if (!res.ok) return { success: false };
    // Some older handlers accidentally return plain text; guard JSON parsing.
    try {
      return (await res.json()) as { success: boolean };
    } catch {
      return { success: false };
    }
  };

  const primary = await tryFetch("/check-ai-binding");
  if (primary.success) return primary;
  return await tryFetch("/check-open-ai-key");
})();

function HasOpenAIKey() {
  const hasOpenAiKey = use(hasOpenAiKeyPromise);

  if (!hasOpenAiKey.success) {
    return (
      <div className="fixed top-0 left-0 right-0 z-50 bg-red-500/10 backdrop-blur-sm">
        <div className="max-w-3xl mx-auto p-4">
          <div className="bg-white dark:bg-neutral-900 rounded-lg shadow-lg border border-red-200 dark:border-red-900 p-4">
            <div className="flex items-start gap-3">
              <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-full">
                <svg
                  className="w-5 h-5 text-red-600 dark:text-red-400"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-labelledby="warningIcon"
                >
                  <title id="warningIcon">Warning Icon</title>
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-red-600 dark:text-red-400 mb-2">
                  Workers AI Binding Not Configured
                </h3>
                <p className="text-neutral-600 dark:text-neutral-300 mb-1">
                  Requests from this UI will not work until the Workers AI binding is available.
                </p>
                <p className="text-neutral-600 dark:text-neutral-300">
                  Please configure the Workers AI binding in your <code className="bg-red-100 dark:bg-red-900/30 px-1.5 py-0.5 rounded text-red-600 dark:text-red-400 font-mono text-sm">wrangler.jsonc</code> as:
                  <br />
                  <code className="bg-red-100 dark:bg-red-900/30 px-1.5 py-0.5 rounded text-red-600 dark:text-red-400 font-mono text-sm">{"\"ai\": {\"binding\": \"AI\"}"}</code>
                  <br />
                  and ensure your generated types include <code className="bg-red-100 dark:bg-red-900/30 px-1.5 py-0.5 rounded text-red-600 dark:text-red-400 font-mono text-sm">AI: Ai</code> in <code className="bg-red-100 dark:bg-red-900/30 px-1.5 py-0.5 rounded text-red-600 dark:text-red-400 font-mono text-sm">env.d.ts</code>.
                  <br />
                  You can also use a different model provider by following these{" "}
                  <a
                    href="https://github.com/cloudflare/agents-starter?tab=readme-ov-file#use-a-different-ai-model-provider"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-red-600 dark:text-red-400"
                  >
                    instructions.
                  </a>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
  return null;
}
