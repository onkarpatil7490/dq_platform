export interface Column {
  name: string;
  type: string;
  totalValues: number;
  uniqueValues: number;
  nullCount: number;
}

export interface TableData {
  name: string;
  columns: Column[];
  rows: Record<string, any>[];
}

export interface Rule {
  id: number;
  name: string;
  table_name: string;
  column_name: string;
  rule: string;
  rule_category: 'info' | 'warning' | 'error';
  sql_query_usr: string;
  sql_query_val: string;
}

export interface ValidationResult {
  goodRows: number;
  badRows: number;
  percentage: number;
  rowValidation: Record<number, boolean>;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface RuleSuggestion {
  rule: string;
  description: string;
  category: 'info' | 'warning' | 'error';
}