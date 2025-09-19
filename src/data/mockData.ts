// Mock data for the application
// This file contains sample data that simulates real database content
// TODO: Replace with actual API calls to fetch real data
// API endpoints needed:
// - GET /api/tables - fetch list of tables
// - GET /api/tables/{table_name}/data - fetch table data and metadata
// - GET /api/rules - fetch existing rules

import { TableData, Rule } from '@/types';

// Available tables in the system
export const availableTables = [
  { name: "meter_data", description: "Meter reading data with customer information" },
  // TODO: Add more tables from your database here
  // API Integration Point: Fetch from GET /get_table_list or similar endpoint
];

export const mockTableData: TableData = {
  name: 'meter_data',
  columns: [
    { name: 'id', type: 'INTEGER', total_values: 1000, unique_values: 1000, null_count: 0 },
    { name: 'meter_id', type: 'VARCHAR', total_values: 1000, unique_values: 850, null_count: 0 },
    { name: 'reading_date', type: 'DATE', total_values: 1000, unique_values: 365, null_count: 5 },
    { name: 'consumption', type: 'DECIMAL', total_values: 1000, unique_values: 950, null_count: 12 },
    { name: 'pincode', type: 'VARCHAR', total_values: 1000, unique_values: 45, null_count: 8 },
    { name: 'customer_type', type: 'VARCHAR', total_values: 1000, unique_values: 4, null_count: 2 },
    { name: 'status', type: 'VARCHAR', total_values: 1000, unique_values: 3, null_count: 0 },
    { name: 'temperature', type: 'DECIMAL', total_values: 1000, unique_values: 180, null_count: 15 },
  ],
  rows: Array.from({ length: 50 }, (_, i) => [
    i + 1,
    `MTR_${String(i + 1).padStart(4, '0')}`,
    new Date(2024, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toISOString().split('T')[0],
    (Math.random() * 500 + 100).toFixed(2),
    ['110001', '110002', '110003', '110004', '110005'][Math.floor(Math.random() * 5)],
    ['Residential', 'Commercial', 'Industrial', 'Government'][Math.floor(Math.random() * 4)],
    ['Active', 'Inactive', 'Suspended'][Math.floor(Math.random() * 3)],
    (Math.random() * 40 + 10).toFixed(1),
  ])
};

export const mockRules: Rule[] = [
  {
    id: 1,
    rule: "Consumption should be between 0 and 1000",
    table_name: "meter_data", 
    column_name: "consumption",
    rule_category: "error",
    sql_query_usr: "SELECT * FROM meter_data WHERE consumption BETWEEN 0 AND 1000",
    sql_query_val: "SELECT row_number() OVER() as row_num FROM meter_data WHERE consumption BETWEEN 0 AND 1000",
    created_at: "2024-01-15T10:30:00Z"
  },
  {
    id: 2,
    rule: "Pincode should not be null",
    table_name: "meter_data",
    column_name: "pincode", 
    rule_category: "warning",
    sql_query_usr: "SELECT * FROM meter_data WHERE pincode IS NOT NULL",
    sql_query_val: "SELECT row_number() OVER() as row_num FROM meter_data WHERE pincode IS NOT NULL",
    created_at: "2024-01-16T14:20:00Z"
  },
  {
    id: 3,
    rule: "Customer type should be valid",
    table_name: "meter_data",
    column_name: "customer_type",
    rule_category: "error", 
    sql_query_usr: "SELECT * FROM meter_data WHERE customer_type IN ('Residential', 'Commercial', 'Industrial', 'Government')",
    sql_query_val: "SELECT row_number() OVER() as row_num FROM meter_data WHERE customer_type IN ('Residential', 'Commercial', 'Industrial', 'Government')",
    created_at: "2024-01-17T09:45:00Z"
  }
];

export const mockRuleSuggestions = [
  {
    rule: 'Check for null values in this column',
    description: 'Ensure data completeness by identifying missing values',
    category: 'warning' as const
  },
  {
    rule: 'Validate data format consistency',
    description: 'Check if all values follow the expected format pattern',
    category: 'error' as const
  },
  {
    rule: 'Identify outliers in numeric data',
    description: 'Detect values that fall outside normal ranges',
    category: 'info' as const
  }
];