import type { Citation } from "../types";

interface CitationLinkProps {
  citation: Citation;
}

export default function CitationLink({ citation }: CitationLinkProps) {
  const scorePct = Math.round(Math.max(0, Math.min(1, citation.score)) * 100);
  const snippetPreview =
    citation.snippet.length > 150
      ? `${citation.snippet.slice(0, 150)}…`
      : citation.snippet;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3 transition-colors hover:border-blue-500 hover:bg-slate-800">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="rounded bg-blue-600 px-2 py-0.5 text-xs font-semibold text-white">
            {citation.filing_ticker}
          </span>
          <span className="text-xs font-medium text-slate-300">
            {citation.filing_form_type}
          </span>
        </div>
        <span className="text-xs font-semibold text-green-400">{scorePct}%</span>
      </div>

      <div className="mt-2 text-xs text-slate-400">
        <span className="text-slate-300">Page {citation.page_number}</span>
      </div>

      <p className="mt-2 text-xs leading-relaxed text-slate-400">{snippetPreview}</p>
    </div>
  );
}
