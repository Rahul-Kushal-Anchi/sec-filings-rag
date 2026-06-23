import CitationLink from "./CitationLink";
import type { Citation } from "../types";

interface RetrievalPanelProps {
  citations: Citation[];
}

export default function RetrievalPanel({ citations }: RetrievalPanelProps) {
  return (
    <aside className="flex h-full w-full flex-col border-l border-slate-800 bg-slate-900/50">
      <div className="border-b border-slate-800 px-4 py-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
          Sources
        </h2>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {citations.length === 0 ? (
          <p className="text-sm text-slate-500">No sources yet</p>
        ) : (
          citations.map((citation, index) => (
            <CitationLink key={`${citation.filing_ticker}-${index}`} citation={citation} />
          ))
        )}
      </div>
    </aside>
  );
}
