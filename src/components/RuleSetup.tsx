import React, { useState } from 'react';
import { Star, Code, CheckCircle, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Rule, RuleSuggestion, ValidationResult } from '@/types';
import { cn } from '@/lib/utils';
import { mockRuleSuggestions } from '@/data/mockData';

interface RuleSetupProps {
  selectedColumn?: string;
  tableName: string;
  onRuleValidate: (rule: string) => Promise<ValidationResult>;
  onRuleSubmit: (rule: Partial<Rule>) => void;
  onSuggestionRequest: () => void;
  initialRule?: Partial<Rule>;
  isCompact?: boolean;
}

export function RuleSetup({ 
  selectedColumn, 
  tableName, 
  onRuleValidate, 
  onRuleSubmit,
  onSuggestionRequest,
  initialRule,
  isCompact = false 
}: RuleSetupProps) {
  const [rule, setRule] = useState(initialRule?.rule || '');
  const [ruleCategory, setRuleCategory] = useState<'info' | 'warning' | 'error'>(
    initialRule?.rule_category || 'info'
  );
  const [sqlQuery, setSqlQuery] = useState(initialRule?.sql_query_usr || '');
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleConvertToSQL = async () => {
    if (!rule.trim() || !selectedColumn) return;
    
    // TODO: Replace with actual API call to convert rule to SQL
    // API endpoint: POST /api/rules/convert-to-sql
    // Payload: { rule, table_name: tableName, column_name: selectedColumn }
    
    // Mock SQL conversion - replace with actual API response
    const mockSQL = `SELECT * FROM ${tableName} WHERE ${selectedColumn} ${
      rule.toLowerCase().includes('not null') ? 'IS NOT NULL' :
      rule.toLowerCase().includes('between') ? 'BETWEEN 0 AND 1000' :
      rule.toLowerCase().includes('valid') ? `IN ('value1', 'value2')` :
      '-- Generated SQL query here'
    }`;
    setSqlQuery(mockSQL);
  };

  const handleValidate = async () => {
    if (!rule.trim()) return;
    
    setIsValidating(true);
    try {
      const result = await onRuleValidate(rule);
      setValidationResult(result);
    } catch (error) {
      console.error('Validation failed:', error);
    } finally {
      setIsValidating(false);
    }
  };

  const handleSubmit = () => {
    if (!rule.trim() || !selectedColumn) return;

    onRuleSubmit({
      name: `${selectedColumn} Rule`,
      table_name: tableName,
      column_name: selectedColumn,
      rule,
      rule_category: ruleCategory,
      sql_query_usr: sqlQuery,
      sql_query_val: `SELECT COUNT(*) FROM ${tableName} WHERE NOT (${rule})`
    });

    // Reset form
    setRule('');
    setSqlQuery('');
    setValidationResult(null);
  };

  const handleSuggestionClick = (suggestion: RuleSuggestion) => {
    setRule(suggestion.rule);
    setRuleCategory(suggestion.category);
    setShowSuggestions(false);
  };

  const getRuleCategoryColor = (category: string) => {
    switch (category) {
      case 'error': return 'text-error bg-error/20 border-error/30';
      case 'warning': return 'text-warning bg-warning/20 border-warning/30';
      case 'info': return 'text-info bg-info/20 border-info/30';
      default: return '';
    }
  };

  return (
    <Card className={cn(
      "bg-gradient-surface border-border shadow-card transition-all duration-300",
      isCompact && "h-fit"
    )}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center justify-between text-foreground">
          <span>Rule Setup</span>
          {selectedColumn && (
            <Badge variant="outline" className="bg-primary/20 text-primary border-primary/30">
              {selectedColumn}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Rule Input */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-foreground">Rule Description</label>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  onSuggestionRequest();
                  setShowSuggestions(true);
                }}
                className="text-accent hover:text-accent-glow"
                disabled={!selectedColumn}
              >
                <Star className="h-4 w-4" />
                AI Suggestions
              </Button>
            </div>
          </div>
          
          <Textarea
            placeholder={
              selectedColumn 
                ? `Describe a data quality rule for the ${selectedColumn} column...`
                : "Select a column first to create a rule"
            }
            value={rule}
            onChange={(e) => setRule(e.target.value)}
            className="min-h-[100px] bg-input border-border focus:border-primary"
            disabled={!selectedColumn}
          />
        </div>

        {/* AI Suggestions */}
        {showSuggestions && (
          <Card className="bg-card-secondary border-border-secondary">
            <CardContent className="p-4 space-y-2">
              <div className="text-sm font-medium text-foreground mb-3">AI Suggestions:</div>
              {mockRuleSuggestions.map((suggestion, index) => (
                <div
                  key={index}
                  className="p-3 rounded-lg bg-background-secondary border border-border cursor-pointer hover:bg-card-secondary transition-colors"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  <div className="font-medium text-foreground mb-1">{suggestion.rule}</div>
                  <div className="text-xs text-muted-foreground">{suggestion.description}</div>
                  <Badge 
                    className={cn("mt-2", getRuleCategoryColor(suggestion.category))}
                    variant="outline"
                  >
                    {suggestion.category}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Rule Type Selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Rule Type</label>
          <Select value={ruleCategory} onValueChange={(value: any) => setRuleCategory(value)}>
            <SelectTrigger className="bg-input border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="info">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-info"></div>
                  Info
                </div>
              </SelectItem>
              <SelectItem value="warning">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-warning"></div>
                  Warning
                </div>
              </SelectItem>
              <SelectItem value="error">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-error"></div>
                  Error
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* SQL Query Display */}
        {sqlQuery && (
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Generated SQL</label>
            <div className="bg-background-tertiary border border-border rounded-lg p-3">
              <code className="text-sm text-accent font-mono">{sqlQuery}</code>
            </div>
          </div>
        )}

        {/* Validation Result */}
        {validationResult && (
          <Card className="bg-card-secondary border-border-secondary">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-foreground">Validation Result</span>
                <Badge 
                  className={cn(
                    validationResult.percentage >= 80 
                      ? "text-success bg-success/20 border-success/30" 
                      : "text-warning bg-warning/20 border-warning/30"
                  )}
                  variant="outline"
                >
                  {validationResult.percentage.toFixed(1)}% Pass
                </Badge>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="text-success">Good Rows: {validationResult.goodRows}</div>
                <div className="text-error">Bad Rows: {validationResult.badRows}</div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4">
          <Button
            variant="outline"
            onClick={handleConvertToSQL}
            disabled={!rule.trim() || !selectedColumn}
            className="flex-1"
          >
            <Code className="h-4 w-4" />
            Convert to SQL
          </Button>
          
          <Button
            variant="info"
            onClick={handleValidate}
            disabled={!rule.trim() || !selectedColumn || isValidating}
            className="flex-1"
          >
            <CheckCircle className="h-4 w-4" />
            {isValidating ? 'Validating...' : 'Validate'}
          </Button>
        </div>

        <Button
          variant="glow"
          onClick={handleSubmit}
          disabled={!rule.trim() || !selectedColumn}
          className="w-full"
          size="lg"
        >
          <Send className="h-4 w-4" />
          Submit Rule
        </Button>
      </CardContent>
    </Card>
  );
}