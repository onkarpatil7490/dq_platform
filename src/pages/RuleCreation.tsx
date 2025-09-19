import React, { useState, useEffect } from 'react';
import { DataTable } from '@/components/DataTable';
import { RuleSetup } from '@/components/RuleSetup';
import { ChatPanel, ChatTrigger } from '@/components/ChatPanel';
import { ValidationResult, RuleSuggestion, Rule, TableData } from '@/types';
import { mockTableData, availableTables } from '@/data/mockData';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { Card } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';

export default function RuleCreation() {
  const [selectedTable, setSelectedTable] = useState<string>();
  const [selectedColumn, setSelectedColumn] = useState<string>();
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidationResult>();
  const [currentTableData, setCurrentTableData] = useState<TableData>(mockTableData);
  const { toast } = useToast();

  // Load table data when table selection changes
  useEffect(() => {
    if (selectedTable) {
      loadTableData(selectedTable);
    }
  }, [selectedTable]);

  const loadTableData = async (tableName: string) => {
    // TODO: API Integration Point - Replace with actual API call
    // API endpoint: POST /get_table_data/
    // Payload: { table_name: tableName, offset: 0, limit: 100 }
    
    try {
      // Mock implementation - replace with actual API call
      console.log(`Loading data for table: ${tableName}`);
      // For now, use mock data
      setCurrentTableData(mockTableData);
      setSelectedColumn(undefined); // Reset column selection
      setValidationResult(undefined); // Reset validation
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to load table data",
        variant: "destructive"
      });
    }
  };

  const handleColumnSelect = (columnName: string) => {
    setSelectedColumn(columnName);
    setValidationResult(undefined);
    toast({
      title: "Column Selected",
      description: `Now creating rules for the "${columnName}" column`,
    });
  };

  const handleConvertToSQL = async (rule: string): Promise<string> => {
    if (!selectedTable || !selectedColumn) {
      toast({
        title: "Missing Selection",
        description: "Please select both a table and column first",
        variant: "destructive"
      });
      return "";
    }

    // TODO: API Integration Point - Replace with actual API call
    // API endpoint: POST /convert_rule_to_sql/
    // Payload: { rule, table_name: selectedTable, column_name: selectedColumn }
    
    try {
      console.log('Converting rule to SQL:', { rule, selectedTable, selectedColumn });
      
      // Mock implementation - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      const mockSQL = `SELECT * FROM ${selectedTable} WHERE ${selectedColumn} IS NOT NULL`;
      
      toast({
        title: "Rule Converted",
        description: "Rule has been converted to SQL successfully",
      });
      
      return mockSQL;
    } catch (error) {
      toast({
        title: "Conversion Failed", 
        description: "Failed to convert rule to SQL",
        variant: "destructive"
      });
      return "";
    }
  };

  const handleRuleValidate = async (sqlQuery: string): Promise<ValidationResult> => {
    if (!selectedTable || !selectedColumn) {
      toast({
        title: "Missing Selection",
        description: "Please select both a table and column first",
        variant: "destructive"
      });
      return { total_rows: 0, total_good_rows: 0, percentage_good_rows: 0, list_good_rows: [] };
    }

    // TODO: API Integration Point - Replace with actual API call
    // API endpoint: POST /validate_sql_query/
    // Payload: { sql_query: sqlQuery, table_name: selectedTable, column_name: selectedColumn }
    
    try {
      console.log('Validating SQL:', { sqlQuery, selectedTable, selectedColumn });
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock validation result - replace with actual API response
      const totalRows = currentTableData.rows.length;
      const goodRows = Math.floor(totalRows * (0.7 + Math.random() * 0.25));
      
      // Create mock row validation
      const rowValidation: Record<number, boolean> = {};
      for (let i = 0; i < totalRows; i++) {
        rowValidation[i] = i < goodRows;
      }
      
      const result = {
        total_rows: totalRows,
        total_good_rows: goodRows,
        percentage_good_rows: (goodRows / totalRows) * 100,
        list_good_rows: Array.from({length: goodRows}, (_, i) => i),
        rowValidation
      };
      
      setValidationResult(result);
      
      toast({
        title: "Validation Complete",
        description: `${result.percentage_good_rows.toFixed(1)}% of rows pass the validation`,
        variant: result.percentage_good_rows >= 80 ? "default" : "destructive"
      });
      
      return result;
    } catch (error) {
      toast({
        title: "Validation Failed",
        description: "Failed to validate SQL query",
        variant: "destructive"
      });
      return { total_rows: 0, total_good_rows: 0, percentage_good_rows: 0, list_good_rows: [] };
    }
  };

  const handleRuleSubmit = async (rule: Partial<Rule>) => {
    if (!selectedTable || !selectedColumn) {
      toast({
        title: "Missing Selection",
        description: "Please select both a table and column first",
        variant: "destructive"
      });
      return;
    }

    // TODO: API Integration Point - Replace with actual API call
    // API endpoint: PUT /add_rule/
    // Payload: { rule: rule.rule, table_name: selectedTable, column_name: selectedColumn, 
    //           rule_category: rule.rule_category, sql_query_usr: rule.sql_query_usr, 
    //           sql_query_val: rule.sql_query_val }
    
    try {
      console.log('Submitting rule:', { ...rule, selectedTable, selectedColumn });
      
      toast({
        title: "Rule Submitted",
        description: `Rule for "${selectedColumn}" has been saved successfully`,
      });
      
      // Reset selections
      setSelectedColumn(undefined);
      setValidationResult(undefined);
    } catch (error) {
      toast({
        title: "Submission Failed",
        description: "Failed to save rule",
        variant: "destructive"
      });
    }
  };

  const handleSuggestionRequest = async () => {
    if (!selectedTable || !selectedColumn) {
      toast({
        title: "Missing Selection",
        description: "Please select both a table and column first to get AI suggestions",
        variant: "destructive"
      });
      return;
    }
    
    // TODO: API Integration Point - Replace with actual API call
    // API endpoint: POST /get_rule_suggestion/
    // Payload: { table_name: selectedTable, column_name: selectedColumn, existing_rules: [] }
    
    try {
      console.log('Getting AI suggestions for:', { selectedTable, selectedColumn });
      toast({
        title: "AI Suggestions",
        description: `Getting suggestions for the "${selectedColumn}" column`,
      });
    } catch (error) {
      toast({
        title: "Suggestion Failed",
        description: "Failed to get AI suggestions",
        variant: "destructive"
      });
    }
  };

  const handleRuleSuggestion = (suggestion: RuleSuggestion) => {
    toast({
      title: "Rule Suggestion Applied",
      description: suggestion.rule,
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Main Content */}
      <div className={cn(
        "transition-all duration-300 ease-in-out",
        isChatOpen ? "mr-96" : "mr-0"
      )}>
        <div className="max-w-7xl mx-auto p-6">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-foreground mb-2">Rule Creation</h1>
            <p className="text-muted-foreground">
              Create data quality rules by selecting tables, columns and defining validation criteria
            </p>
          </div>

          {/* Table Selection Section */}
          <Card className="mb-6 p-6">
            <div className="space-y-4">
              <div>
                <Label htmlFor="table-select" className="text-sm font-medium">
                  Select Table
                </Label>
                <Select value={selectedTable} onValueChange={setSelectedTable}>
                  <SelectTrigger className="w-full mt-1">
                    <SelectValue placeholder="Choose a table to work with" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableTables.map((table) => (
                      <SelectItem key={table.name} value={table.name}>
                        <div>
                          <div className="font-medium">{table.name}</div>
                          <div className="text-xs text-muted-foreground">{table.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {selectedTable && (
                <div className="text-sm text-muted-foreground">
                  Selected table: <span className="font-medium text-foreground">{selectedTable}</span>
                  {selectedColumn && (
                    <span> → Column: <span className="font-medium text-foreground">{selectedColumn}</span></span>
                  )}
                </div>
              )}
            </div>
          </Card>

          {/* Split Layout */}
          <div className="grid lg:grid-cols-2 gap-6 h-[calc(100vh-350px)]">
            {/* Data Table Section */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-foreground">
                  {selectedTable ? `${selectedTable} Table` : "Select a Table"}
                </h2>
                {selectedTable && (
                  <div className="text-sm text-muted-foreground">
                    {currentTableData.rows.length} rows × {currentTableData.columns.length} columns
                  </div>
                )}
              </div>
              
              {selectedTable ? (
                <DataTable
                  data={currentTableData}
                  selectedColumn={selectedColumn}
                  onColumnSelect={handleColumnSelect}
                  validationResult={validationResult}
                  isCompact={isChatOpen}
                />
              ) : (
                <Card className="h-96 flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-muted-foreground mb-2">No table selected</div>
                    <div className="text-sm text-muted-foreground">
                      Please select a table above to view its data
                    </div>
                  </div>
                </Card>
              )}
            </div>

            {/* Rule Setup Section */}
            <div className={cn(
              "transition-all duration-300",
              isChatOpen && "transform -translate-x-48"
            )}>
              <div className="mb-4">
                <h2 className="text-xl font-semibold text-foreground">Rule Configuration</h2>
                <p className="text-sm text-muted-foreground">
                  {selectedTable && selectedColumn 
                    ? `Creating rule for "${selectedColumn}" column in "${selectedTable}" table`
                    : selectedTable 
                      ? "Select a column to start creating rules"
                      : "Select a table and column to start creating rules"
                  }
                </p>
              </div>
              
              <RuleSetup
                selectedColumn={selectedColumn}
                tableName={selectedTable}
                onConvertToSQL={handleConvertToSQL}
                onRuleValidate={handleRuleValidate}
                onRuleSubmit={handleRuleSubmit}
                onSuggestionRequest={handleSuggestionRequest}
                isCompact={isChatOpen}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Chat Components */}
      <ChatTrigger onClick={() => setIsChatOpen(true)} />
      <ChatPanel
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        onRuleSuggestion={handleRuleSuggestion}
        selectedColumn={selectedColumn}
      />
    </div>
  );
}