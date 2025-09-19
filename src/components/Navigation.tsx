import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Database, Settings, Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export function Navigation() {
  const location = useLocation();

  const navItems = [
    {
      href: '/',
      label: 'Rule Creation',
      icon: Plus,
      description: 'Create new data quality rules'
    },
    {
      href: '/management',
      label: 'Rule Management',
      icon: Settings,
      description: 'Manage existing rules'
    }
  ];

  return (
    <nav className="bg-background-secondary border-b border-border">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="p-2 rounded-lg bg-gradient-primary shadow-glow group-hover:shadow-elevated transition-all duration-300">
              <Database className="h-6 w-6 text-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">DataQuality Pro</h1>
              <p className="text-xs text-muted-foreground">Rule Management Dashboard</p>
            </div>
          </Link>

          {/* Navigation Items */}
          <div className="flex items-center space-x-1">
            {navItems.map((item) => {
              const isActive = location.pathname === item.href;
              const Icon = item.icon;

              return (
                <Button
                  key={item.href}
                  asChild
                  variant={isActive ? "default" : "ghost"}
                  className={cn(
                    "gap-2 transition-all duration-300",
                    isActive && "shadow-glow"
                  )}
                >
                  <Link to={item.href}>
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </Link>
                </Button>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}