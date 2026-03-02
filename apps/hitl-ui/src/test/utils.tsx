/**
 * Test Utilities
 * Common utilities and helpers for testing React components
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { type RenderOptions, render } from '@testing-library/react';
import type { ReactElement } from 'react';
import { BrowserRouter } from 'react-router-dom';

/**
 * Custom render function that includes common providers
 * Use this instead of the plain render from @testing-library/react
 */
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
    initialRoute?: string;
    queryClient?: QueryClient;
}

export function renderWithProviders(
    ui: ReactElement,
    {
        initialRoute = '/',
        queryClient = new QueryClient({
            defaultOptions: {
                queries: {
                    retry: false, // Disable retries in tests
                    gcTime: 0, // Disable caching in tests
                },
            },
        }),
        ...renderOptions
    }: CustomRenderOptions = {}
) {
    // Set initial route if needed
    window.history.pushState({}, 'Test page', initialRoute);

    function Wrapper({ children }: { children: React.ReactNode }) {
        return (
            <BrowserRouter>
                <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
            </BrowserRouter>
        );
    }

    return {
        ...render(ui, { wrapper: Wrapper, ...renderOptions }),
        queryClient,
    };
}

/**
 * Helper to create a mock fetch response
 */
export function createMockResponse<T>(data: T, status = 200): Response {
    return {
        ok: status >= 200 && status < 300,
        status,
        json: async () => data,
        text: async () => JSON.stringify(data),
        headers: new Headers(),
    } as Response;
}

/**
 * Helper to wait for async updates
 */
export const waitForAsync = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * Mock local storage for tests
 */
export function createLocalStorageMock() {
    let store: Record<string, string> = {};

    return {
        getItem: (key: string) => store[key] || null,
        setItem: (key: string, value: string) => {
            store[key] = value;
        },
        removeItem: (key: string) => {
            delete store[key];
        },
        clear: () => {
            store = {};
        },
        get length() {
            return Object.keys(store).length;
        },
        key: (index: number) => Object.keys(store)[index] || null,
    };
}

// Re-export everything from @testing-library/react
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
