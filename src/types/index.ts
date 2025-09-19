// Backend API Types
export interface Column {
  name: string;
  type: string;
  total_values?: number;
  unique_values?: number;
  null_count?: number;
}

export interface TableData {
  name: string;
  columns: Column[];
  rows: any[][];
}

export interface Rule {
  id: number;
  rule: string;
  table_name: string;
  column_name: string;
  rule_category: 'info' | 'warning' | 'error';
  sql_query_usr: string;
  sql_query_val: string;
  created_at?: string;
}

export interface ValidationResult {
  total_rows: number;
  total_good_rows: number;
  percentage_good_rows: number;
  list_good_rows: number[];
  rowValidation?: Record<number, boolean>; // Frontend computed field
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

// API Request/Response Types
export interface ConvertRuleRequest {
  table_name: string;
  column_name: string;
  rule: string;
}

export interface ValidateSQLRequest {
  sql_query: string;
  table_name: string;
  column_name: string;
}

export interface AddRuleRequest {
  rule: string;
  table_name: string;
  column_name: string;
  rule_category: string;
  sql_query_usr: string;
  sql_query_val: string;
}

export interface RuleSuggestionRequest {
  table_name: string;
  column_name: string;
  existing_rules?: string[];
}

export interface TableDataRequest {
  table_name: string;
  offset?: number;
  limit?: number;
}

export interface InfoRequest {
  table_name: string;
  column_name: string;
}

export interface ChatbotRequest {
  user_input: string;
  table_name: string;
  column_name: string;
}