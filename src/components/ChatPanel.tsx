import React, { useState, useRef, useEffect } from 'react';
import { Send, X, MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ChatMessage, RuleSuggestion } from '@/types';
import { cn } from '@/lib/utils';

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onRuleSuggestion: (suggestion: RuleSuggestion) => void;
  selectedColumn?: string;
}

export function ChatPanel({ isOpen, onClose, onRuleSuggestion, selectedColumn }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: `Hello! I'm here to help you create data quality rules. ${
        selectedColumn 
          ? `I see you've selected the "${selectedColumn}" column. What would you like to validate about this column?`
          : 'Select a column to get started with rule suggestions.'
      }`,
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // TODO: Replace with actual API call to AI chat service
    // API endpoint: POST /api/ai/chat
    // Payload: { message: input, context: { selectedColumn, tableName } }
    
    // Simulate AI response
    setTimeout(() => {
      const suggestions = [
        {
          rule: `Check for null values in ${selectedColumn || 'selected column'}`,
          description: 'Ensure data completeness',
          category: 'warning' as const
        },
        {
          rule: `Validate ${selectedColumn || 'column'} format consistency`,
          description: 'Check data format patterns',
          category: 'error' as const
        }
      ];

      const aiResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Here are some suggestions for the ${selectedColumn || 'selected column'} column:`,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
    }, 1500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-gradient-surface border-l border-border shadow-elevated z-50 transform transition-transform duration-300">
      <Card className="h-full rounded-none border-0 bg-transparent">
        <CardHeader className="flex flex-row items-center justify-between pb-4 border-b border-border">
          <CardTitle className="flex items-center gap-2 text-foreground">
            <MessageCircle className="h-5 w-5 text-primary" />
            AI Assistant
          </CardTitle>
          <Button variant="ghost" size="icon-sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>

        <CardContent className="p-0 flex flex-col h-[calc(100vh-80px)]">
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex",
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  )}
                >
                  <div
                    className={cn(
                      "max-w-[80%] rounded-lg px-4 py-2 text-sm",
                      message.role === 'user'
                        ? "bg-primary text-primary-foreground"
                        : "bg-card-secondary text-foreground border border-border"
                    )}
                  >
                    {message.content}
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-card-secondary text-foreground border border-border rounded-lg px-4 py-2 text-sm">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          <div className="p-4 border-t border-border">
            <div className="flex gap-2">
              <Input
                placeholder="Ask about data quality rules..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                className="bg-input border-border"
              />
              <Button
                onClick={handleSendMessage}
                disabled={!input.trim()}
                size="icon"
                variant="default"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export function ChatTrigger({ onClick }: { onClick: () => void }) {
  return (
    <Button
      onClick={onClick}
      variant="glow"
      size="icon-lg"
      className="fixed bottom-6 right-6 z-40 shadow-elevated hover:shadow-glow"
    >
      <MessageCircle className="h-6 w-6" />
    </Button>
  );
}