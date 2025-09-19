# Data Quality Rule Management Dashboard - Project Structure

## Main Entry Point
**File**: `src/main.tsx`
- This is the main entry point that renders the React application
- Imports: App component from `src/App.tsx`

## Application Structure

### Core Application (`src/App.tsx`)
- Sets up routing, providers, and global components
- **Imports**:
  - `Navigation` from `@/components/Navigation`
  - Page components from `@/pages/`
  - UI components for toasts and tooltips

### Pages
Located in `src/pages/`:

#### 1. Rule Creation (`RuleCreation.tsx`)
- **Purpose**: Main interface for creating data quality rules
- **Key Features**:
  - 50/50 split layout (Data Table + Rule Setup)
  - Column selection and validation
  - AI suggestions integration
  - Chat panel functionality
- **Imports**:
  - `DataTable` from `@/components/DataTable`
  - `RuleSetup` from `@/components/RuleSetup`
  - `ChatPanel, ChatTrigger` from `@/components/ChatPanel`
  - Types from `@/types/index`
  - Mock data from `@/data/mockData`

#### 2. Rule Management (`RuleManagement.tsx`)
- **Purpose**: View, edit, and manage existing rules
- **Key Features**:
  - Filter by table and column
  - Rule list with edit/delete actions
  - Integration with Rule Creation page
- **Imports**:
  - UI components (Button, Card, Badge, etc.)
  - Types and mock data

### Components
Located in `src/components/`:

#### Main Components:
1. **DataTable.tsx**
   - Displays tabular data with column selection
   - Shows column metadata (type, values, nulls)
   - Highlights validation results
   - **Imports**: UI components, types, utilities

2. **RuleSetup.tsx**
   - Rule creation form interface
   - SQL conversion and validation
   - AI suggestions display
   - **Imports**: UI components, types, mock data

3. **ChatPanel.tsx**
   - AI chat interface for rule assistance
   - Sliding panel with conversation
   - **Imports**: UI components, types, mock data

4. **Navigation.tsx**
   - Top navigation bar
   - Route switching between pages
   - **Imports**: UI components, React Router

#### UI Components (`src/components/ui/`)
- Shadcn/ui components library
- Pre-built, customizable components
- Examples: Button, Card, Dialog, Input, etc.

### Data & Types
1. **Types (`src/types/index.ts`)**
   - TypeScript interfaces for:
     - Rule, TableData, ValidationResult
     - RuleSuggestion, ChatMessage

2. **Mock Data (`src/data/mockData.ts`)**
   - Sample table data and rules
   - AI suggestions and chat responses
   - **Note**: Contains API integration comments

### Styling & Configuration
1. **index.css**: Global styles and design system tokens
2. **tailwind.config.ts**: Tailwind CSS configuration with custom colors
3. **lib/utils.ts**: Utility functions (cn for className merging)

### Key Dependencies
- **React + TypeScript**: Core framework
- **React Router**: Navigation
- **Tailwind CSS**: Styling
- **Radix UI**: Accessible UI primitives
- **Lucide React**: Icons
- **React Query**: Data fetching (setup for future API integration)

## API Integration Points

The following files contain TODO comments indicating where to integrate with your FastAPI backend:

1. **Rule Creation (`src/pages/RuleCreation.tsx`)**:
   - `handleRuleValidate()`: POST /api/rules/validate
   - `handleRuleSubmit()`: POST /api/rules
   - `handleSuggestionRequest()`: POST /api/ai/suggestions

2. **Rule Management (`src/pages/RuleManagement.tsx`)**:
   - `loadRules()`: GET /api/rules
   - `handleEdit()`: GET /api/rules/{id}
   - `handleDelete()`: DELETE /api/rules/{id}

3. **Data Loading (`src/data/mockData.ts`)**:
   - Replace mock data with actual API calls
   - GET /api/tables
   - GET /api/tables/{table_name}/data

4. **Chat Panel (`src/components/ChatPanel.tsx`)**:
   - `sendMessage()`: POST /api/ai/chat

## Development Workflow
1. Start with `npm run dev` or `bun dev`
2. Main development happens in `src/` directory
3. UI components are in `src/components/ui/`
4. Pages are in `src/pages/`
5. Modify `src/data/mockData.ts` to add more sample data
6. Update `src/types/index.ts` when adding new data structures