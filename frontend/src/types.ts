export interface Citation {
  filing_ticker: string;
  filing_form_type: string;
  filing_date: string;
  page_number: number;
  snippet: string;
  score: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  timestamp: Date;
}
