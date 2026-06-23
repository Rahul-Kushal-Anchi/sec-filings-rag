import { useCallback, useState } from "react";

import type { Citation } from "../types";

interface AskResult {
  text: string;
  citations: Citation[];
}

interface QueryResponse {
  text: string;
  citations: Citation[];
  confidence: number;
  latency_ms: number;
}

const DEFAULT_BACKEND_URL = "http://localhost:8000";

/**
 * Hook that posts a question to the backend /v1/query endpoint and returns
 * the answer text plus citations. Uses the synchronous JSON path (stream=false)
 * for simplicity; SSE streaming is a stretch goal.
 */
export function useQueryStream(backendUrl: string = DEFAULT_BACKEND_URL) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ask = useCallback(
    async (question: string): Promise<AskResult> => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(`${backendUrl}/v1/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, max_results: 5 }),
        });

        if (!response.ok) {
          throw new Error(
            `Request failed with status ${response.status} ${response.statusText}`,
          );
        }

        const data: QueryResponse = await response.json();
        return {
          text: data.text,
          citations: data.citations ?? [],
        };
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Unknown error contacting backend";
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [backendUrl],
  );

  return { ask, isLoading, error };
}
