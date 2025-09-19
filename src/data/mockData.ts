// Mock data for the application
// This file contains sample data that simulates real database content
// TODO: Replace with actual API calls to fetch real data
// API endpoints needed:
// - GET /api/tables - fetch list of tables
// - GET /api/tables/{table_name}/data - fetch table data and metadata
// - GET /api/rules - fetch existing rules

import { TableData, Rule } from '@/types';

export const mockTableData: TableData = {
  name: 'meter_data',
  columns: [
    { name: 'id', type: 'INTEGER', totalValues: 1000, uniqueValues: 1000, nullCount: 0 },
    { name: 'meter_id', type: 'VARCHAR', totalValues: 1000, uniqueValues: 850, nullCount: 0 },
    { name: 'reading_date', type: 'DATE', totalValues: 1000, uniqueValues: 365, nullCount: 5 },
    { name: 'consumption', type: 'DECIMAL', totalValues: 1000, uniqueValues: 950, nullCount: 12 },
    { name: 'pincode', type: 'VARCHAR', totalValues: 1000, uniqueValues: 45, nullCount: 8 },
    { name: 'customer_type', type: 'VARCHAR', totalValues: 1000, uniqueValues: 4, nullCount: 2 },
    { name: 'status', type: 'VARCHAR', totalValues: 1000, uniqueValues: 3, nullCount: 0 },
    { name: 'temperature', type: 'DECIMAL', totalValues: 1000, uniqueValues: 180, nullCount: 15 },
  ],
  rows: Array.from({ length: 50 }, (_, i) => ({
    id: i + 1,
    meter_id: `MTR_${String(i + 1).padStart(4, '0')}`,
    reading_date: new Date(2024, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1).toISOString().split('T')[0],
    consumption: (Math.random() * 500 + 100).toFixed(2),
    pincode: ['110001', '110002', '110003', '110004', '110005'][Math.floor(Math.random() * 5)],
    customer_type: ['Residential', 'Commercial', 'Industrial', 'Government'][Math.floor(Math.random() * 4)],
    status: ['Active', 'Inactive', 'Suspended'][Math.floor(Math.random() * 3)],
    temperature: (Math.random() * 40 + 10).toFixed(1),
  }))
};

export const mockRules: Rule[] = [
  {
    id: 1,
    name: 'Consumption Range Check',
    table_name: 'meter_data',
    column_name: 'consumption',
    rule: 'Consumption should be between 0 and 1000',
    rule_category: 'error',
    sql_query_usr: 'SELECT * FROM meter_data WHERE consumption BETWEEN 0 AND 1000',
    sql_query_val: 'SELECT COUNT(*) FROM meter_data WHERE consumption NOT BETWEEN 0 AND 1000'
  },
  {
    id: 2,
    name: 'Pincode Not Null',
    table_name: 'meter_data',
    column_name: 'pincode',
    rule: 'Pincode should not be null',
    rule_category: 'warning',
    sql_query_usr: 'SELECT * FROM meter_data WHERE pincode IS NOT NULL',
    sql_query_val: 'SELECT COUNT(*) FROM meter_data WHERE pincode IS NULL'
  },
  {
    id: 3,
    name: 'Valid Customer Type',
    table_name: 'meter_data',
    column_name: 'customer_type',
    rule: 'Customer type should be one of: Residential, Commercial, Industrial, Government',
    rule_category: 'error',
    sql_query_usr: 'SELECT * FROM meter_data WHERE customer_type IN ("Residential", "Commercial", "Industrial", "Government")',
    sql_query_val: 'SELECT COUNT(*) FROM meter_data WHERE customer_type NOT IN ("Residential", "Commercial", "Industrial", "Government")'
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