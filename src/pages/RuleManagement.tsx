import React, { useState } from 'react';
import { Edit, Trash2, Plus, Filter } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { Rule } from '@/types';
import { mockRules, mockTableData } from '@/data/mockData';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { useNavigate } from 'react-router-dom';

export default function RuleManagement() {
  const [rules, setRules] = useState<Rule[]>(mockRules);
  const [selectedTable, setSelectedTable] = useState<string>('all');
  const [selectedColumn, setSelectedColumn] = useState<string>('all');
  const [selectedRuleType, setSelectedRuleType] = useState<string>('all');
  const { toast } = useToast();
  const navigate = useNavigate();

  const filteredRules = rules.filter((rule) => {
    if (selectedTable !== 'all' && rule.table_name !== selectedTable) return false;
    if (selectedColumn !== 'all' && rule.column_name !== selectedColumn) return false;
    if (selectedRuleType !== 'all' && rule.rule_category !== selectedRuleType) return false;
    return true;
  });

  const handleEditRule = (rule: Rule) => {
    // TODO: Replace with actual API call to fetch rule details
    // API endpoint: GET /api/rules/{id}
    // Navigate to rule creation page with rule data
    navigate('/', { state: { editRule: rule } });
  };

  const handleDeleteRule = (ruleId: number) => {
    // TODO: Replace with actual API call to delete rule
    // API endpoint: DELETE /api/rules/{id}
    setRules(prev => prev.filter(rule => rule.id !== ruleId));
    toast({
      title: "Rule Deleted",
      description: "The rule has been successfully removed",
    });
  };

  const handleAddNewRule = () => {
    const state: any = {};
    if (selectedColumn !== 'all') {
      state.selectedColumn = selectedColumn;
    }
    navigate('/', { state });
  };

  const getRuleCategoryColor = (category: string) => {
    switch (category) {
      case 'error': return 'text-error bg-error/20 border-error/30';
      case 'warning': return 'text-warning bg-warning/20 border-warning/30';
      case 'info': return 'text-info bg-info/20 border-info/30';
      default: return '';
    }
  };

  const getRuleCategoryIcon = (category: string) => {
    switch (category) {
      case 'error': return '🔴';
      case 'warning': return '🟡';
      case 'info': return '🔵';
      default: return '⚪';
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Rule Management</h1>
          <p className="text-muted-foreground">
            Manage and monitor your data quality rules across all tables
          </p>
        </div>

        {/* Filters */}
        <Card className="mb-6 bg-gradient-surface border-border shadow-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <Filter className="h-5 w-5" />
              Filters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Table</label>
                <Select value={selectedTable} onValueChange={setSelectedTable}>
                  <SelectTrigger className="bg-input border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Tables</SelectItem>
                    <SelectItem value="meter_data">meter_data</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Column</label>
                <Select value={selectedColumn} onValueChange={setSelectedColumn}>
                  <SelectTrigger className="bg-input border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Columns</SelectItem>
                    {mockTableData.columns.map((column) => (
                      <SelectItem key={column.name} value={column.name}>
                        {column.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Rule Type</label>
                <Select value={selectedRuleType} onValueChange={setSelectedRuleType}>
                  <SelectTrigger className="bg-input border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="info">Info</SelectItem>
                    <SelectItem value="warning">Warning</SelectItem>
                    <SelectItem value="error">Error</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-end">
                <Button
                  onClick={handleAddNewRule}
                  variant="glow"
                  className="w-full"
                >
                  <Plus className="h-4 w-4" />
                  Add New Rule
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Rules List */}
        <Card className="bg-gradient-surface border-border shadow-card">
          <CardHeader>
            <CardTitle className="flex items-center justify-between text-foreground">
              <span>Data Quality Rules</span>
              <Badge variant="outline" className="bg-primary/20 text-primary border-primary/30">
                {filteredRules.length} rules
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {filteredRules.length === 0 ? (
              <div className="p-8 text-center">
                <div className="text-muted-foreground mb-4">No rules found matching your filters</div>
                <Button onClick={handleAddNewRule} variant="outline">
                  <Plus className="h-4 w-4" />
                  Create First Rule
                </Button>
              </div>
            ) : (
              <div className="divide-y divide-border">
                {filteredRules.map((rule) => (
                  <div
                    key={rule.id}
                    className="p-6 hover:bg-card-secondary/50 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 space-y-3">
                        {/* Rule Header */}
                        <div className="flex items-center gap-3">
                          <span className="text-lg">{getRuleCategoryIcon(rule.rule_category)}</span>
                          <h3 className="text-lg font-semibold text-foreground">{rule.name}</h3>
                          <Badge 
                            className={cn("text-xs", getRuleCategoryColor(rule.rule_category))}
                            variant="outline"
                          >
                            {rule.rule_category}
                          </Badge>
                        </div>

                        {/* Rule Details */}
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Table:</span>
                            <span className="ml-2 text-foreground font-medium">{rule.table_name}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Column:</span>
                            <span className="ml-2 text-accent font-medium">{rule.column_name}</span>
                          </div>
                        </div>

                        {/* Rule Description */}
                        <div className="text-sm">
                          <span className="text-muted-foreground">Rule:</span>
                          <p className="mt-1 text-foreground">{rule.rule}</p>
                        </div>

                        {/* SQL Query */}
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="ghost" size="sm" className="text-xs p-0 h-auto">
                              View SQL Query
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="max-w-2xl">
                            <DialogHeader>
                              <DialogTitle>SQL Query - {rule.name}</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4">
                              <div>
                                <label className="text-sm font-medium text-muted-foreground">Display Query:</label>
                                <div className="mt-2 p-3 bg-background-tertiary border border-border rounded-lg">
                                  <code className="text-sm text-accent font-mono">{rule.sql_query_usr}</code>
                                </div>
                              </div>
                              <div>
                                <label className="text-sm font-medium text-muted-foreground">Validation Query:</label>
                                <div className="mt-2 p-3 bg-background-tertiary border border-border rounded-lg">
                                  <code className="text-sm text-accent font-mono">{rule.sql_query_val}</code>
                                </div>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>
                      </div>

                      {/* Action Buttons */}
                      <div className="flex items-center gap-2 ml-4">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleEditRule(rule)}
                        >
                          <Edit className="h-4 w-4" />
                          Edit
                        </Button>
                        
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button variant="destructive" size="sm">
                              <Trash2 className="h-4 w-4" />
                              Delete
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>Delete Rule</AlertDialogTitle>
                              <AlertDialogDescription>
                                Are you sure you want to delete "{rule.name}"? This action cannot be undone.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                              <AlertDialogAction
                                onClick={() => handleDeleteRule(rule.id)}
                                className="bg-error hover:bg-error-glow"
                              >
                                Delete
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}