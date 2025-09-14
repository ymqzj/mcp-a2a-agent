
"use client";
import React, { useState, useRef, useEffect } from "react";
import { Textarea, Button, Select, ScrollArea, Avatar } from "@mantine/core";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypePrism from 'rehype-prism-plus';

type Msg = { role: "user" | "assistant"; content: string };

export default function Page() {
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [sending, setSending] = useState(false);
  const [model, setModel] = useState("gpt-4o");
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [msgs]);

  const ask = async () => {
    if (!input.trim() || sending) return;
    const user = input.trim();
    setMsgs(m => [...m, { role: "user", content: user }]);
    setInput("");
    setSending(true);
    try {
      const res = await fetch((process.env.NEXT_PUBLIC_BACKEND_URL || "") + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: user, model })
      });
      const data = await res.json();
      const answer = data?.answer ?? JSON.stringify(data ?? {});
      setMsgs(m => [...m, { role: "assistant", content: answer }]);
    } catch (err) {
      setMsgs(m => [...m, { role: "assistant", content: "请求失败，请稍后重试。" }]);
    } finally {
      setSending(false);
    }
  };

  const onKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  };

  const clearChat = () => setMsgs([]);

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 20 }}>
      <div style={{ display: 'flex', gap: 20 }}>
        <aside style={{ width: 260, background: 'white', borderRadius: 8, padding: 16, display: 'flex', flexDirection: 'column', gap: 12, height: '80vh' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Avatar color="indigo">AI</Avatar>
            <div>
              <h4 style={{ margin: 0 }}>AI Agent</h4>
              <p style={{ margin: 0, fontSize: 12, color: '#6b7280' }}>MCP + CrewAI 演示</p>
            </div>
          </div>

          <Button variant="subtle" mt="md" onClick={clearChat} style={{ width: '100%' }}>
            + 新会话
          </Button>

          <div style={{ marginTop: 12 }}>
            <p style={{ margin: '6px 0 8px 0', fontSize: 12, color: '#6b7280' }}>模型</p>
            <Select value={model} onChange={(v) => setModel(v || "gpt-4o")} data={[{ value: 'gpt-4o', label: 'gpt-4o' }, { value: 'gpt-4o-mini', label: 'gpt-4o-mini' }, { value: 'gpt-3.5-turbo', label: 'gpt-3.5-turbo' }]} />
          </div>

          <p style={{ fontSize: 12, color: '#6b7280', marginTop: 'auto' }}>Enter 发送 · Shift+Enter 换行</p>
        </aside>

        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '80vh', background: 'white', borderRadius: 8, overflow: 'hidden' }}>
          <div style={{ padding: 16, borderBottom: '1px solid rgba(0,0,0,0.06)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontWeight: 600 }}>会话</div>
            <div style={{ color: '#6b7280' }}>与 MCP 后端交互</div>
          </div>

          <div style={{ padding: 20, flex: 1, overflow: 'auto' }} ref={containerRef as any}>
            {msgs.length === 0 && (
              <div style={{ textAlign: 'center', color: '#9ca3af' }}>还没有消息 — 请输入问题并回车发送</div>
            )}

            {msgs.map((m, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
                <div className={`msg-bubble ${m.role === 'user' ? 'user' : 'agent'} fade-in`}>
                  <div className="msg-meta">{m.role === 'user' ? '你' : 'Agent'}</div>
                  <div style={{ whiteSpace: 'pre-wrap' }}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypePrism]}>{m.content}</ReactMarkdown>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="input-bar">
            <div className="input-textarea" style={{ flex: 1 }}>
              <Textarea
                value={input}
                onChange={e => setInput((e.target as HTMLTextAreaElement).value)}
                onKeyDown={onKeyDown}
                minRows={2}
                autosize
                placeholder="在这里输入消息... (Enter 发送，Shift+Enter 换行)"
                style={{ width: '100%' }}
                disabled={sending}
              />
            </div>

            <button className="send-btn" onClick={ask} disabled={sending}>
              {sending ? '发送中...' : '发送'}
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}
