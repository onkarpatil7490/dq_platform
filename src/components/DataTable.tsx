import React, { useState } from 'react';
import { Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { TableData, Column, ValidationResult } from '@/types';
import { cn } from '@/lib/utils';

interface DataTableProps {
  data: TableData;
  selectedColumn?: string;
  onColumnSelect: (columnName: string) => void;
  validationResult?: ValidationResult;
  isCompact?: boolean;
}

export function DataTable({ 
  data, 
  selectedColumn, 
  onColumnSelect, 
  validationResult,
  isCompact = false 
}: DataTableProps) {
  const [hoveredColumn, setHoveredColumn] = useState<string | null>(null);

  const getCellStyle = (columnName: string, rowIndex: number) => {
    if (!validationResult) return '';
    
    const isValid = validationResult.rowValidation?.[rowIndex];
    if (columnName === selectedColumn && isValid !== undefined) {
      return isValid ? 'bg-success/20 text-success-glow' : 'bg-error/20 text-error-glow';
    }
    return '';
  };

  const getColumnHeaderStyle = (columnName: string) => {
    return cn(
      'transition-all duration-300',
      selectedColumn === columnName && 'bg-primary/20 text-primary-glow shadow-glow',
      hoveredColumn === columnName && 'bg-card-secondary'
    );
  };

  return (
    <Card className={cn(
      "bg-gradient-surface border-border shadow-card transition-all duration-300",
      isCompact && "h-[60vh]"
    )}>
      <CardContent className="p-0">
        <div className={cn(
          "overflow-auto",
          isCompact ? "max-h-[60vh]" : "max-h-[80vh]"
        )}>
          <table className="w-full">
            <thead className="sticky top-0 bg-background-secondary/95 backdrop-blur-sm border-b border-border">
              <tr>
                {data.columns.map((column) => (
                  <th
                    key={column.name}
                    className={cn(
                      "px-4 py-3 text-left font-medium text-foreground border-r border-border last:border-r-0 cursor-pointer select-none",
                      getColumnHeaderStyle(column.name)
                    )}
                    onMouseEnter={() => setHoveredColumn(column.name)}
                    onMouseLeave={() => setHoveredColumn(null)}
                    onClick={() => onColumnSelect(column.name)}
                  >
                    <div className="flex items-center gap-2">
                      <span className="truncate">{column.name}</span>
                      <ColumnInfoButton column={column} />
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, rowIndex) => (
                <tr
                  key={rowIndex}
                  className={cn(
                    "border-b border-border/50 hover:bg-card-secondary/50 transition-colors",
                    rowIndex % 2 === 0 && "bg-background-secondary/30"
                  )}
                >
                  {data.columns.map((column) => (
                    <td
                      key={column.name}
                      className={cn(
                        "px-4 py-2 text-sm border-r border-border/30 last:border-r-0 transition-all duration-300",
                        getCellStyle(column.name, rowIndex)
                      )}
                    >
                      <div className="truncate max-w-[200px]">
                        {row[data.columns.indexOf(column)] ?? (
                          <span className="text-muted-foreground italic">null</span>
                        )}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function ColumnInfoButton({ column }: { column: Column }) {
  const nullPercentage = column.null_count && column.total_values 
    ? ((column.null_count / column.total_values) * 100).toFixed(1)
    : '0.0';
  const uniquePercentage = column.unique_values && column.total_values
    ? ((column.unique_values / column.total_values) * 100).toFixed(1)
    : '0.0';

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon-sm"
          className="h-5 w-5 text-muted-foreground hover:text-info-glow hover:bg-info/20"
          onClick={(e) => e.stopPropagation()}
        >
          <Info className="h-3 w-3" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64 bg-card border-border shadow-elevated">
        <div className="space-y-3">
          <div className="font-medium text-foreground">{column.name}</div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Type:</span>
              <span className="text-accent font-medium">{column.type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Total Values:</span>
              <span className="text-foreground">{column.total_values?.toLocaleString() || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Unique Values:</span>
              <span className="text-foreground">
                {column.unique_values?.toLocaleString() || 'N/A'} ({uniquePercentage}%)
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Null Count:</span>
              <span className={cn(
                "font-medium",
                (column.null_count || 0) > 0 ? "text-warning" : "text-success"
              )}>
                {(column.null_count || 0).toLocaleString()} ({nullPercentage}%)
              </span>
            </div>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}