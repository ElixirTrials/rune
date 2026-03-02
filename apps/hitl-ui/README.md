# HITL UI

## Purpose
React/Vite application for Human-in-the-Loop workflows. This is where users approve, edit, or reject AI-generated content.

## Development

1.  **Install Dependencies**:
    ```bash
    npm install
    ```
2.  **Start Dev Server**:
    ```bash
    npm run dev
    ```

## Architecture
- **Features**: Code is organized by feature (e.g., `src/features/dashboard`, `src/features/users`).
- **State Management**:
    - **Server State**: `TanStack Query` (React Query) for fetching from API.
    - **Client State**: `Zustand` for local UI state.
- **UI Components**: `shadcn/ui` (in `src/components/ui`).

## Adding a New Page
1.  Create the page component in `src/screens/`.
2.  Add the route in `src/App.tsx` (using `react-router-dom`).
3.  Create feature-specific components in `src/features/<feature>/`.
