import { useEffect, useRef } from "react";

import type { Message } from "../types";

interface ChatPanelProps {
  messages: Message[];
  isLoading: boolean;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export default function ChatPanel({ messages, isLoading }: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 space-y-4 overflow-y-auto p-6">
      {messages.length === 0 && !isLoading && (
        <div className="flex h-full items-center justify-center">
          <p className="text-slate-600">
            Ask a question to get started, e.g. &ldquo;What are the main risk factors?&rdquo;
          </p>
        </div>
      )}

      {messages.map((message) => {
        const isUser = message.role === "user";
        return (
          <div
            key={message.id}
            className={`flex ${isUser ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                isUser
                  ? "rounded-br-sm bg-blue-600 text-white"
                  : "rounded-bl-sm bg-slate-800 text-slate-100"
              }`}
            >
              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {message.content}
              </p>
              <span
                className={`mt-1 block text-[10px] ${
                  isUser ? "text-blue-200" : "text-slate-500"
                }`}
              >
                {formatTime(message.timestamp)}
              </span>
            </div>
          </div>
        );
      })}

      {isLoading && (
        <div className="flex justify-start">
          <div className="flex items-center gap-2 rounded-2xl rounded-bl-sm bg-slate-800 px-4 py-3">
            <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
