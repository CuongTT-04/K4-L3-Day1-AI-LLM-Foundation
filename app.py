import json
import os
import sys
import time
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Import logic from solution/solution.py
try:
    from solution.solution import (
        OPENAI_MODEL,
        OPENAI_MINI_MODEL,
        count_tokens,
        estimate_cost,
        PRICING_PER_1K_TOKENS,
        retry_with_backoff,
    )
except ImportError:
    from template import (
        OPENAI_MODEL,
        OPENAI_MINI_MODEL,
        count_tokens,
        estimate_cost,
        PRICING_PER_1K_TOKENS,
        retry_with_backoff,
    )

app = FastAPI(title="AI Assistant Web UI", description="Modern React + Tailwind UI for CLI Assistant")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    persona: Optional[str] = "Bạn là trợ giảng thân thiện của khóa học AI LLM, trả lời ngắn gọn, súc tích bằng tiếng Việt."
    model: Optional[str] = OPENAI_MODEL
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512

@app.get("/api/config")
def get_config():
    return {
        "default_model": OPENAI_MODEL,
        "mini_model": OPENAI_MINI_MODEL,
        "pricing": PRICING_PER_1K_TOKENS,
    }

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    full_messages = [{"role": "system", "content": req.persona}] if req.persona else []
    for m in req.messages[-6:]:  # Maintain context limit matching CLI assistant
        full_messages.append({"role": m.role, "content": m.content})

    user_query = req.messages[-1].content if req.messages else ""

    def event_stream():
        start_time = time.perf_counter()
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model=req.model or OPENAI_MODEL,
                messages=full_messages,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_reply += delta
                    payload = json.dumps({"type": "chunk", "delta": delta})
                    yield f"data: {payload}\n\n"

            latency = round(time.perf_counter() - start_time, 2)
            cost_info = estimate_cost(user_query, full_reply, req.model or OPENAI_MODEL)
            in_tokens = count_tokens(user_query, req.model or OPENAI_MODEL)
            out_tokens = count_tokens(full_reply, req.model or OPENAI_MODEL)

            done_payload = json.dumps({
                "type": "done",
                "latency": latency,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
                "cost": cost_info.get("total_cost", 0.0),
            })
            yield f"data: {done_payload}\n\n"

        except Exception as e:
            err_payload = json.dumps({"type": "error", "error": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


HTML_CONTENT = """<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Assistant Pro — LLM Foundation</title>
  
  <!-- Tailwind CSS v3 via Play CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#f0f4ff',
              100: '#dbe4fe',
              500: '#3b82f6',
              600: '#2563eb',
              700: '#1d4ed8',
              800: '#1e40af',
              900: '#1e3a8a',
              950: '#0b132b',
            },
            surface: {
              800: '#131b2e',
              850: '#0e1626',
              900: '#0a0f1d',
              950: '#060913',
            }
          },
          fontFamily: {
            sans: ['Inter', 'system-ui', 'sans-serif'],
            mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
          }
        }
      }
    }
  </script>

  <!-- Google Fonts: Inter & JetBrains Mono -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

  <!-- React 18 & Babel & Marked -->
  <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

  <style>
    body {
      font-family: 'Inter', sans-serif;
      background-color: #060913;
      color: #e2e8f0;
    }
    .custom-scrollbar::-webkit-scrollbar {
      width: 6px;
    }
    .custom-scrollbar::-webkit-scrollbar-track {
      background: transparent;
    }
    .custom-scrollbar::-webkit-scrollbar-thumb {
      background: #1e293b;
      border-radius: 9999px;
    }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover {
      background: #334155;
    }
    .glass-panel {
      background: rgba(14, 22, 38, 0.75);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-card {
      background: rgba(19, 27, 46, 0.6);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    /* Typing cursor animation */
    .typing-cursor::after {
      content: '▋';
      display: inline-block;
      vertical-align: baseline;
      animation: blink 0.9s infinite;
      color: #60a5fa;
      margin-left: 2px;
    }
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }
    /* Markdown rendering styles */
    .prose-custom p { margin-bottom: 0.75rem; line-height: 1.65; }
    .prose-custom p:last-child { margin-bottom: 0; }
    .prose-custom pre {
      background: #0b1120;
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 0.5rem;
      padding: 0.75rem 1rem;
      overflow-x: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      margin: 0.75rem 0;
    }
    .prose-custom code {
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255,255,255,0.1);
      padding: 0.15rem 0.35rem;
      border-radius: 0.25rem;
      font-size: 0.85em;
    }
    .prose-custom pre code {
      background: transparent;
      padding: 0;
    }
    .prose-custom ul { list-style-type: disc; padding-left: 1.25rem; margin-bottom: 0.75rem; }
    .prose-custom ol { list-style-type: decimal; padding-left: 1.25rem; margin-bottom: 0.75rem; }
    .prose-custom blockquote {
      border-left: 3px solid #3b82f6;
      padding-left: 0.75rem;
      color: #94a3b8;
      font-style: italic;
      margin: 0.75rem 0;
    }
  </style>
</head>
<body class="h-screen flex overflow-hidden">
  <div id="root" class="w-full h-full"></div>

  <script type="text/babel">
    const { useState, useEffect, useRef } = React;

    const Icons = {
      cpu: (
        <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>
          <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>
        </svg>
      ),
      chevronLeft: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7"/>
        </svg>
      ),
      menu: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      ),
      zap: (
        <svg className="w-3.5 h-3.5 text-amber-400" fill="currentColor" viewBox="0 0 24 24">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
      ),
      trash: (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
        </svg>
      ),
      bot: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4M8 16h.01M16 16h.01"/>
        </svg>
      ),
      botLarge: (
        <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4M8 16h.01M16 16h.01"/>
        </svg>
      ),
      user: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/>
        </svg>
      ),
      send: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      ),
      sparkle: (
        <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4M4 19h4m5-16l2.286 6.857L21 12l-5.714 2.286L13 21l-2.286-6.857L5 12l5.714-2.286L13 3z"/>
        </svg>
      ),
      clock: (
        <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
        </svg>
      ),
    };

    const PERSONAS = [
      {
        id: "ta",
        name: "Trợ Giảng AI (Mặc định Lab)",
        prompt: "Bạn là trợ giảng thân thiện của khóa học AI LLM, trả lời ngắn gọn, súc tích bằng tiếng Việt, kèm ví dụ thực tế nếu có thể.",
        badge: "Khuyên dùng"
      },
      {
        id: "dev",
        name: "Senior Fullstack Engineer",
        prompt: "Bạn là Senior Fullstack Developer giàu kinh nghiệm. Hãy trả lời chuyên sâu, tập trung vào code sạch, kiến trúc phần mềm, performance và giải pháp tối ưu.",
        badge: "Technical"
      },
      {
        id: "fin",
        name: "Chuyên Gia Tài Chính",
        prompt: "Bạn là chuyên gia tài chính và kinh tế học. Hãy phân tích chuyên sâu, sử dụng thuật ngữ tài chính chuẩn mực và dẫn chứng logic chặt chẽ.",
        badge: "Finance"
      },
      {
        id: "writer",
        name: "Biên Tập Viên Sáng Tạo",
        prompt: "Bạn là chuyên gia sáng tạo nội dung và viết lách. Hãy diễn đạt truyền cảm, mượt mà, cuốn hút với vốn từ phong phú.",
        badge: "Creative"
      }
    ];

    const SUGGESTIONS = [
      "Giải thích khác biệt giữa temperature và top_p trong 1 câu",
      "Tại sao tiếng Việt tốn nhiều token hơn tiếng Anh khi gọi LLM?",
      "So sánh Time to First Token (TTFT) giữa Streaming và Non-streaming",
      "Viết hàm Python tính token và ước lượng chi phí gọi GPT-4o"
    ];

    function App() {
      const [messages, setMessages] = useState([]);
      const [input, setInput] = useState("");
      const [isStreaming, setIsStreaming] = useState(false);
      const [sidebarOpen, setSidebarOpen] = useState(true);

      // LLM Config
      const [persona, setPersona] = useState(PERSONAS[0].prompt);
      const [selectedPersonaId, setSelectedPersonaId] = useState("ta");
      const [model, setModel] = useState("openai/gpt-oss-120b");
      const [temperature, setTemperature] = useState(0.7);
      const [topP, setTopP] = useState(0.9);
      const [maxTokens, setMaxTokens] = useState(512);

      // Session Metrics
      const [sessionStats, setSessionStats] = useState({
        turns: 0,
        totalTokens: 0,
        totalCost: 0,
        lastLatency: 0,
      });

      const messagesEndRef = useRef(null);
      const textareaRef = useRef(null);

      useEffect(() => {
        // Fetch config from server
        fetch("/api/config")
          .then(res => res.json())
          .then(data => {
            if (data.default_model) setModel(data.default_model);
          })
          .catch(err => console.error("Config fetch error:", err));
      }, []);

      useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }, [messages, isStreaming]);

      // Auto resize textarea
      useEffect(() => {
        if (textareaRef.current) {
          textareaRef.current.style.height = "auto";
          textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 140) + "px";
        }
      }, [input]);

      const handleSend = async (customPrompt = null) => {
        const textToSend = customPrompt || input;
        if (!textToSend.trim() || isStreaming) return;

        const newMessages = [...messages, { role: "user", content: textToSend }];
        setMessages(newMessages);
        setInput("");
        setIsStreaming(true);

        // Placeholder for assistant response
        const assistantIndex = newMessages.length;
        setMessages(prev => [...prev, { role: "assistant", content: "", latency: null }]);

        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              messages: newMessages,
              persona: persona,
              model: model,
              temperature: parseFloat(temperature),
              top_p: parseFloat(topP),
              max_tokens: parseInt(maxTokens),
            }),
          });

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let accumulated = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const textChunk = decoder.decode(value);
            const lines = textChunk.split("\\n");

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                try {
                  const data = JSON.parse(line.slice(6));
                  if (data.type === "chunk") {
                    accumulated += data.delta;
                    setMessages(prev => {
                      const updated = [...prev];
                      updated[assistantIndex] = {
                        ...updated[assistantIndex],
                        content: accumulated,
                      };
                      return updated;
                    });
                  } else if (data.type === "done") {
                    setMessages(prev => {
                      const updated = [...prev];
                      updated[assistantIndex] = {
                        ...updated[assistantIndex],
                        latency: data.latency,
                        tokens: data.total_tokens,
                        cost: data.cost,
                      };
                      return updated;
                    });

                    setSessionStats(prev => ({
                      turns: prev.turns + 1,
                      totalTokens: prev.totalTokens + data.total_tokens,
                      totalCost: prev.totalCost + data.cost,
                      lastLatency: data.latency,
                    }));
                  } else if (data.type === "error") {
                    accumulated += `\\n\\n⚠️ **Lỗi:** ${data.error}`;
                    setMessages(prev => {
                      const updated = [...prev];
                      updated[assistantIndex] = {
                        ...updated[assistantIndex],
                        content: accumulated,
                      };
                      return updated;
                    });
                  }
                } catch (e) {
                  // Ignore JSON parse error on incomplete chunks
                }
              }
            }
          }
        } catch (err) {
          setMessages(prev => {
            const updated = [...prev];
            updated[assistantIndex] = {
              ...updated[assistantIndex],
              content: `⚠️ Không thể kết nối với server: ${err.message}`,
            };
            return updated;
          });
        } finally {
          setIsStreaming(false);
        }
      };

      const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleSend();
        }
      };

      const selectPersona = (p) => {
        setSelectedPersonaId(p.id);
        setPersona(p.prompt);
      };

      const clearChat = () => {
        setMessages([]);
        setInput("");
        setSessionStats({
          turns: 0,
          totalTokens: 0,
          totalCost: 0,
          lastLatency: 0,
        });
      };

      return (
        <div className="flex h-screen w-screen overflow-hidden bg-[#060913]">
          {/* SIDEBAR PANEL */}
          <aside
            className={`transition-all duration-300 ease-in-out border-r border-white/5 flex flex-col z-20 ${
              sidebarOpen ? "w-80" : "w-0 -translate-x-full"
            } glass-panel overflow-hidden`}
          >
            {/* Sidebar Header */}
            <div className="p-4 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
                  {Icons.cpu}
                </div>
                <div>
                  <h2 className="font-semibold text-sm text-white tracking-wide">Cấu Hình Trợ Lý</h2>
                  <span className="text-[11px] text-blue-400 font-mono">Day 1 • LLM Foundation</span>
                </div>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="text-slate-400 hover:text-white p-1 rounded-md hover:bg-white/5 transition"
                title="Đóng sidebar"
              >
                {Icons.chevronLeft}
              </button>
            </div>

            {/* Config controls */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6 text-xs">
              {/* Persona selection */}
              <div>
                <label className="block text-slate-400 font-medium mb-2 uppercase tracking-wider text-[10px]">
                  Persona / Vai Trò Trợ Lý
                </label>
                <div className="space-y-2">
                  {PERSONAS.map(p => (
                    <div
                      key={p.id}
                      onClick={() => selectPersona(p)}
                      className={`p-2.5 rounded-lg cursor-pointer transition border ${
                        selectedPersonaId === p.id
                          ? "bg-blue-600/15 border-blue-500/40 text-white"
                          : "bg-white/[0.02] border-white/5 text-slate-300 hover:bg-white/[0.05]"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-[12px]">{p.name}</span>
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30">
                          {p.badge}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-2">{p.prompt}</p>
                    </div>
                  ))}
                </div>

                {/* Custom system prompt input */}
                <div className="mt-3">
                  <span className="text-[11px] text-slate-400 block mb-1">Tùy chỉnh System Prompt:</span>
                  <textarea
                    value={persona}
                    onChange={(e) => {
                      setPersona(e.target.value);
                      setSelectedPersonaId("custom");
                    }}
                    rows={3}
                    className="w-full bg-[#090e1a] border border-white/10 rounded-lg p-2 text-slate-200 text-xs focus:border-blue-500 focus:outline-none custom-scrollbar"
                  />
                </div>
              </div>

              {/* Model selection */}
              <div>
                <label className="block text-slate-400 font-medium mb-2 uppercase tracking-wider text-[10px]">
                  Mô Hình LLM (Model)
                </label>
                <input
                  type="text"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full bg-[#090e1a] border border-white/10 rounded-lg px-2.5 py-2 text-slate-200 font-mono text-[11px] focus:border-blue-500 focus:outline-none"
                  placeholder="e.g. openai/gpt-oss-120b"
                />
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Model lớn: <code>openai/gpt-oss-120b</code> | Model nhỏ: <code>openai/gpt-oss-20b</code>
                </span>
              </div>

              {/* Hyperparameters Sliders */}
              <div className="space-y-4 pt-2 border-t border-white/5">
                {/* Temperature */}
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-300 font-medium">Temperature</span>
                    <span className="font-mono text-blue-400 font-semibold">{temperature}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1.5"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(e.target.value)}
                    className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-slate-500 mt-0.5">
                    <span>Chính xác (0.0)</span>
                    <span>Sáng tạo (1.5)</span>
                  </div>
                </div>

                {/* Top-P */}
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-300 font-medium">Top-P</span>
                    <span className="font-mono text-blue-400 font-semibold">{topP}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.05"
                    value={topP}
                    onChange={(e) => setTopP(e.target.value)}
                    className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                </div>

                {/* Max Tokens */}
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-300 font-medium">Max Tokens</span>
                    <span className="font-mono text-blue-400 font-semibold">{maxTokens}</span>
                  </div>
                  <input
                    type="range"
                    min="64"
                    max="2048"
                    step="64"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(e.target.value)}
                    className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                </div>
              </div>
            </div>

            {/* Sidebar Footer Stats */}
            <div className="p-3 border-t border-white/5 bg-[#090e1a]/70">
              <div className="flex justify-between items-center text-[10px] text-slate-400 mb-1">
                <span>Số lượt chat:</span>
                <span className="font-mono text-slate-200">{sessionStats.turns}</span>
              </div>
              <div className="flex justify-between items-center text-[10px] text-slate-400 mb-1">
                <span>Tổng Token ước tính:</span>
                <span className="font-mono text-slate-200">{sessionStats.totalTokens}</span>
              </div>
              <div className="flex justify-between items-center text-[10px] text-slate-400">
                <span>Chi phí ước tính:</span>
                <span className="font-mono text-emerald-400 font-semibold">
                  ${sessionStats.totalCost.toFixed(5)}
                </span>
              </div>
            </div>
          </aside>

          {/* MAIN CHAT AREA */}
          <main className="flex-1 flex flex-col h-full overflow-hidden bg-gradient-to-b from-[#0a0f1d] to-[#060913]">
            {/* Top Navbar */}
            <header className="h-14 border-b border-white/5 px-4 flex items-center justify-between glass-panel z-10">
              <div className="flex items-center gap-3">
                {!sidebarOpen && (
                  <button
                    onClick={() => setSidebarOpen(true)}
                    className="text-slate-400 hover:text-white p-1.5 rounded-md hover:bg-white/5 transition"
                    title="Mở cài đặt"
                  >
                    {Icons.menu}
                  </button>
                )}
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></div>
                  <span className="font-semibold text-sm text-slate-100">LLM CLI Assistant — Web Edition</span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono">
                    {model}
                  </span>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-2">
                {sessionStats.lastLatency > 0 && (
                  <span className="text-xs font-mono text-slate-400 bg-white/5 px-2.5 py-1 rounded-md border border-white/5 flex items-center gap-1.5">
                    {Icons.zap}
                    {sessionStats.lastLatency}s
                  </span>
                )}
                <button
                  type="button"
                  onClick={clearChat}
                  className="text-xs text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 px-2.5 py-1.5 rounded-md border border-white/5 transition flex items-center gap-1.5 cursor-pointer select-none"
                  title="Xóa đoạn chat và đặt lại phiên"
                >
                  {Icons.trash}
                  <span className="hidden sm:inline font-medium">Làm mới</span>
                </button>
              </div>
            </header>

            {/* Chat Messages Container */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 space-y-5">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto px-4">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-blue-600 to-cyan-500 flex items-center justify-center mb-4 shadow-xl shadow-blue-500/20 ring-1 ring-white/20">
                    {Icons.botLarge}
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    Chào bạn! Trợ lý AI đã sẵn sàng
                  </h3>
                  <p className="text-slate-400 text-xs mb-6 leading-relaxed">
                    Giao diện hiện đại kết nối trực tiếp với logic của <code className="text-blue-300 font-mono">solution.py</code>. 
                    Hỗ trợ streaming thời gian thực, đo độ trễ và ước lượng chi phí chính xác.
                  </p>

                  {/* Suggestion buttons */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full text-left">
                    {SUGGESTIONS.map((s, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSend(s)}
                        className="p-3 rounded-xl glass-card hover:bg-white/[0.08] hover:border-blue-500/40 text-xs text-slate-300 transition text-left flex items-start gap-2 group cursor-pointer"
                      >
                        <div className="shrink-0 mt-0.5 group-hover:scale-110 transition">{Icons.sparkle}</div>
                        <span>{s}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((m, idx) => (
                  <div
                    key={idx}
                    className={`flex gap-3 max-w-3xl ${
                      m.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
                    }`}
                  >
                    {/* Avatar */}
                    <div
                      className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 shadow-md ${
                        m.role === "user"
                          ? "bg-gradient-to-tr from-indigo-600 to-blue-500 text-white"
                          : "bg-slate-800 border border-white/10 text-blue-400"
                      }`}
                    >
                      {m.role === "user" ? Icons.user : Icons.bot}
                    </div>

                    {/* Message Bubble */}
                    <div className="flex flex-col gap-1 max-w-[85%]">
                      <div
                        className={`p-3.5 rounded-2xl text-xs md:text-sm leading-relaxed ${
                          m.role === "user"
                            ? "bg-blue-600 text-white rounded-tr-none shadow-lg shadow-blue-600/10"
                            : "glass-card text-slate-200 rounded-tl-none border-white/10"
                        }`}
                      >
                        {m.role === "assistant" ? (
                          <div
                            className="prose-custom"
                            dangerouslySetInnerHTML={{
                              __html: marked.parse(m.content || "") + (isStreaming && idx === messages.length - 1 ? '<span class="typing-cursor"></span>' : "")
                            }}
                          />
                        ) : (
                          <p className="whitespace-pre-wrap">{m.content}</p>
                        )}
                      </div>

                      {/* Assistant Stats Footer */}
                      {m.role === "assistant" && (m.latency !== null && m.latency !== undefined) && (
                        <div className="flex items-center gap-3 text-[10px] text-slate-500 px-1 font-mono">
                          <span className="flex items-center gap-1">
                            {Icons.clock}
                            {m.latency}s
                          </span>
                          {m.tokens && <span>• {m.tokens} tokens</span>}
                          {m.cost !== undefined && (
                            <span>• ${m.cost.toFixed(6)}</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Bar */}
            <div className="p-4 border-t border-white/5 glass-panel">
              <div className="max-w-3xl mx-auto">
                <div className="relative flex items-end bg-[#090e1a] border border-white/10 rounded-2xl p-1.5 focus-within:border-blue-500/60 focus-within:ring-2 focus-within:ring-blue-500/20 transition shadow-lg">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Nhập tin nhắn cho trợ lý (Enter để gửi, Shift+Enter để xuống dòng)..."
                    rows={1}
                    disabled={isStreaming}
                    className="w-full bg-transparent text-slate-100 text-xs md:text-sm px-3 py-2.5 focus:outline-none resize-none custom-scrollbar max-h-36"
                  />
                  <button
                    onClick={() => handleSend()}
                    disabled={!input.trim() || isStreaming}
                    className={`p-2.5 rounded-xl transition shrink-0 ${
                      input.trim() && !isStreaming
                        ? "bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-500/20 cursor-pointer"
                        : "bg-white/5 text-slate-500 cursor-not-allowed"
                    }`}
                  >
                    {isStreaming ? (
                      <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                      Icons.send
                    )}
                  </button>
                </div>
                <div className="flex justify-between items-center text-[10px] text-slate-500 px-2 mt-2">
                  <span>Khóa K4-L3: LLM API Foundation</span>
                  <span className="font-mono">Context window: 3 lượt chat gần nhất</span>
                </div>
              </div>
            </div>
          </main>
        </div>
      );
    }

    const root = ReactDOM.createRoot(document.getElementById("root"));
    root.render(<App />);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return HTML_CONTENT

if __name__ == "__main__":
    import uvicorn
    print("================================================================")
    print("  AI Assistant Web UI - Khởi động máy chủ thành công!")
    print("  Mở trình duyệt tại địa chỉ: http://127.0.0.1:8000")
    print("================================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000)
