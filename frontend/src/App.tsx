import { useState } from "react";

import ChatPanel from "./components/ChatPanel";
import RetrievalPanel from "./components/RetrievalPanel";
import { useQueryStream } from "./hooks/useQueryStream";
import type { Citation, Message } from "./types";

function createId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [latestCitations, setLatestCitations] = useState<Citation[]>([]);
  const { ask, isLoading, error } = useQueryStream();

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || isLoading) {
      return;
    }

    const userMessage: Message = {
      id: createId(),
      role: "user",
      content: question,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const result = await ask(question);
      const assistantMessage: Message = {
        id: createId(),
        role: "assistant",
        content: result.text,
        citations: result.citations,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setLatestCitations(result.citations);
    } catch {
      // Error is surfaced via the `error` state from the hook; render below.
    }
  };

  return (
    <div className="flex h-screen flex-col bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-xl font-semibold tracking-tight">SEC RAG Assistant</h1>
        <p className="text-sm text-slate-400">
          Ask questions about SEC 10-K and 10-Q filings
        </p>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <main className="flex flex-1 flex-col">
          <ChatPanel messages={messages} isLoading={isLoading} />

          {error && (
            <div className="mx-6 mb-2 rounded-lg border border-red-800 bg-red-950/60 px-4 py-2 text-sm text-red-300">
              {error}
            </div>
          )}

          <form
            onSubmit={handleSubmit}
            className="flex items-center gap-3 border-t border-slate-800 p-4"
          >
            <input
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask a question about the filings…"
              disabled={isLoading}
              className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={isLoading || input.trim().length === 0}
              className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? "Asking…" : "Ask"}
            </button>
          </form>
        </main>

        <div className="hidden w-80 shrink-0 md:block">
          <RetrievalPanel citations={latestCitations} />
        </div>
      </div>
    </div>
  );
}
