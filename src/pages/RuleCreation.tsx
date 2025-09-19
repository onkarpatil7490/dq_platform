import React, { useState } from 'react';
import { DataTable } from '@/components/DataTable';
import { RuleSetup } from '@/components/RuleSetup';
import { ChatPanel, ChatTrigger } from '@/components/ChatPanel';
import { ValidationResult, RuleSuggestion, Rule } from '@/types';
import { mockTableData } from '@/data/mockData';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

export default function RuleCreation() {
  const [selectedColumn, setSelectedColumn] = useState<string>();
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidationResult>();
  const { toast } = useToast();

  const handleColumnSelect = (columnName: string) => {
    setSelectedColumn(columnName);
    setValidationResult(undefined);
    toast({
      title: "Column Selected",
      description: `Now creating rules for the "${columnName}" column`,
    });
  };

  const handleRuleValidate = async (rule: string): Promise<ValidationResult> => {
    // TODO: Replace with actual API call to validate rule
    // API endpoint: POST /api/rules/validate
    // Payload: { rule, table_name: mockTableData.name, column_name: selectedColumn }
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Mock validation result - replace with actual API response
    const totalRows = mockTableData.rows.length;
    const goodRows = Math.floor(totalRows * (0.7 + Math.random() * 0.25));
    const badRows = totalRows - goodRows;
    const percentage = (goodRows / totalRows) * 100;
    
    // Create mock row validation
    const rowValidation: Record<number, boolean> = {};
    for (let i = 0; i < totalRows; i++) {
      rowValidation[i] = i < goodRows;
    }
    
    const result = {
      goodRows,
      badRows,
      percentage,
      rowValidation
    };
    
    setValidationResult(result);
    
    toast({
      title: "Validation Complete",
      description: `${percentage.toFixed(1)}% of rows pass the validation`,
      variant: percentage >= 80 ? "default" : "destructive"
    });
    
    return result;
  };

  const handleRuleSubmit = (rule: Partial<Rule>) => {
    // TODO: Replace with actual API call to save rule
    // API endpoint: POST /api/rules
    // Payload: rule object
    console.log('Submitting rule:', rule);
    toast({
      title: "Rule Submitted",
      description: `Rule for "${rule.column_name}" has been saved successfully`,
    });
    
    // Reset selections
    setSelectedColumn(undefined);
    setValidationResult(undefined);
  };

  const handleSuggestionRequest = () => {
    if (!selectedColumn) {
      toast({
        title: "No Column Selected",
        description: "Please select a column first to get AI suggestions",
        variant: "destructive"
      });
      return;
    }
    
    // TODO: Replace with actual API call to get AI suggestions
    // API endpoint: POST /api/ai/suggestions
    // Payload: { table_name: mockTableData.name, column_name: selectedColumn }
    toast({
      title: "AI Suggestions",
      description: `Getting suggestions for the "${selectedColumn}" column`,
    });
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
              Create data quality rules by selecting columns and defining validation criteria
            </p>
          </div>

          {/* Split Layout */}
          <div className="grid lg:grid-cols-2 gap-6 h-[calc(100vh-200px)]">
            {/* Data Table Section */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-foreground">
                  {mockTableData.name} Table
                </h2>
                <div className="text-sm text-muted-foreground">
                  {mockTableData.rows.length} rows × {mockTableData.columns.length} columns
                </div>
              </div>
              
              <DataTable
                data={mockTableData}
                selectedColumn={selectedColumn}
                onColumnSelect={handleColumnSelect}
                validationResult={validationResult}
                isCompact={isChatOpen}
              />
            </div>

            {/* Rule Setup Section */}
            <div className={cn(
              "transition-all duration-300",
              isChatOpen && "transform -translate-x-48"
            )}>
              <div className="mb-4">
                <h2 className="text-xl font-semibold text-foreground">Rule Configuration</h2>
                <p className="text-sm text-muted-foreground">
                  {selectedColumn 
                    ? `Creating rule for "${selectedColumn}" column`
                    : "Select a column to start creating rules"
                  }
                </p>
              </div>
              
              <RuleSetup
                selectedColumn={selectedColumn}
                tableName={mockTableData.name}
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