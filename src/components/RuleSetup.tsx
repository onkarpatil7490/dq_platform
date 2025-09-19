import React, { useState } from 'react';
import { Star, Code, CheckCircle, Plus, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { ValidationResult, Rule } from '@/types';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface RuleSetupProps {
  selectedColumn?: string;
  tableName?: string;
  onConvertToSQL: (rule: string) => Promise<string>;
  onRuleValidate: (sqlQuery: string) => Promise<ValidationResult>;
  onRuleSubmit: (rule: Partial<Rule>) => void;
  onSuggestionRequest: () => void;
  initialRule?: string;
  isCompact?: boolean;
}

export function RuleSetup({
  selectedColumn,
  tableName,
  onConvertToSQL,
  onRuleValidate,
  onRuleSubmit,
  onSuggestionRequest,
  initialRule = "",
  isCompact = false
}: RuleSetupProps) {
  const [ruleDescription, setRuleDescription] = useState(initialRule);
  const [ruleCategory, setRuleCategory] = useState<'info' | 'warning' | 'error'>('info');
  const [generatedSQL, setGeneratedSQL] = useState<string>();
  const [validationResults, setValidationResults] = useState<ValidationResult>();
  const [isConverting, setIsConverting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const { toast } = useToast();

  const handleConvertToSQL = async () => {
    if (!ruleDescription.trim()) {
      toast({
        title: "No Rule Description",
        description: "Please enter a rule description first",
        variant: "destructive"
      });
      return;
    }

    setIsConverting(true);
    try {
      const sqlQuery = await onConvertToSQL(ruleDescription);
      setGeneratedSQL(sqlQuery);
    } catch (error) {
      console.error('Error converting rule to SQL:', error);
    } finally {
      setIsConverting(false);
    }
  };

  const handleValidate = async () => {
    if (!generatedSQL) {
      toast({
        title: "No SQL Query",
        description: "Please convert the rule to SQL first", 
        variant: "destructive"
      });
      return;
    }

    setIsValidating(true);
    try {
      const results = await onRuleValidate(generatedSQL);
      setValidationResults(results);
    } catch (error) {
      console.error('Error validating rule:', error);
    } finally {
      setIsValidating(false);
    }
  };

  const handleSubmit = () => {
    if (!ruleDescription.trim() || !selectedColumn || !tableName || !generatedSQL) {
      toast({
        title: "Incomplete Information",
        description: "Please complete all steps: rule description, SQL conversion, and validation",
        variant: "destructive"
      });
      return;
    }

    const rule: Partial<Rule> = {
      rule: ruleDescription,
      table_name: tableName,
      column_name: selectedColumn,
      rule_category: ruleCategory,
      sql_query_usr: generatedSQL,
      sql_query_val: generatedSQL
    };

    onRuleSubmit(rule);
    
    // Reset form
    setRuleDescription("");
    setGeneratedSQL(undefined);
    setValidationResults(undefined);
    setRuleCategory('info');
  };

  return (
    <Card className={cn(
      "bg-gradient-surface border-border shadow-card transition-all duration-300",
      isCompact && "h-[60vh]"
    )}>
      <CardHeader className="pb-4">
        <CardTitle className="text-lg font-semibold text-foreground">
          Rule Configuration
        </CardTitle>
      </CardHeader>

      <CardContent className={cn(
        "space-y-4",
        isCompact ? "max-h-[50vh] overflow-y-auto" : "max-h-[70vh] overflow-y-auto"
      )}>
        {!selectedColumn && (
          <Card className="p-4 bg-muted/30 border-warning/50">
            <div className="text-center text-muted-foreground">
              <div className="text-lg mb-1">👆</div>
              <div className="text-sm">Select a column from the table to start creating rules</div>
            </div>
          </Card>
        )}

        {selectedColumn && (
          <>
            {/* Rule Input Section */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="rule-description" className="text-sm font-medium">
                  Rule Description
                </Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onSuggestionRequest}
                  className="text-accent hover:text-accent-glow hover:bg-accent/20"
                  disabled={!selectedColumn}
                >
                  <Star className="h-4 w-4 mr-1" />
                  AI Suggestions
                </Button>
              </div>
              
              <Textarea
                id="rule-description"
                placeholder={`Enter a rule for the "${selectedColumn}" column...`}
                value={ruleDescription}
                onChange={(e) => setRuleDescription(e.target.value)}
                className="min-h-[80px] resize-none"
                disabled={!selectedColumn}
              />
            </div>

            {/* Convert to SQL Button */}
            <Button
              onClick={handleConvertToSQL}
              variant="outline"
              className="w-full flex items-center gap-2"
              disabled={!ruleDescription.trim() || isConverting}
            >
              <Code className="h-4 w-4" />
              {isConverting ? "Converting..." : "Convert to SQL"}
            </Button>

            {/* Generated SQL Display */}
            {generatedSQL && (
              <Card className="p-4 bg-muted/50">
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <Code className="h-4 w-4" />
                  Generated SQL
                </h4>
                <pre className="text-sm bg-background-secondary p-3 rounded border overflow-x-auto">
                  <code className="text-accent">{generatedSQL}</code>
                </pre>
              </Card>
            )}

            {/* Validate Button */}
            <Button
              onClick={handleValidate}
              variant="outline"
              className="w-full flex items-center gap-2"
              disabled={isValidating || !generatedSQL}
            >
              <CheckCircle className="h-4 w-4" />
              {isValidating ? "Validating..." : "Validate"}
            </Button>

            {/* Validation Results */}
            {validationResults && (
              <Card className="p-4 bg-muted/50">
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Validation Results
                </h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-success">{validationResults.total_good_rows}</div>
                    <div className="text-muted-foreground">Good Rows</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-destructive">{validationResults.total_rows - validationResults.total_good_rows}</div>
                    <div className="text-muted-foreground">Bad Rows</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">{validationResults.percentage_good_rows.toFixed(1)}%</div>
                    <div className="text-muted-foreground">Pass Rate</div>
                  </div>
                </div>
              </Card>
            )}

            {/* Rule Category Selection */}
            <div className="space-y-2">
              <Label htmlFor="rule-category" className="text-sm font-medium">
                Rule Type
              </Label>
              <Select value={ruleCategory} onValueChange={(value: 'info' | 'warning' | 'error') => setRuleCategory(value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="info">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-info/20 text-info border-info/30">Info</Badge>
                      <span className="text-sm text-muted-foreground">Informational check</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="warning">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-warning/20 text-warning border-warning/30">Warning</Badge>
                      <span className="text-sm text-muted-foreground">Data quality concern</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="error">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-error/20 text-error border-error/30">Error</Badge>
                      <span className="text-sm text-muted-foreground">Critical data issue</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Submit Button */}
            <Button
              onClick={handleSubmit}
              className="w-full bg-gradient-primary hover:bg-gradient-primary/90"
              disabled={!ruleDescription.trim() || !selectedColumn || !tableName || !generatedSQL || !validationResults}
            >
              <Plus className="h-4 w-4 mr-2" />
              Submit Rule
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}