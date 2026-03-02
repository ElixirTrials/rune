/**
 * Example Test: Component with API Calls
 *
 * This example demonstrates:
 * - Testing async data fetching
 * - Mocking API calls
 * - Testing loading states
 * - Testing error states
 * - Using React Query in tests
 */

import { useQuery } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '../utils';
import { renderWithProviders } from '../utils';

// Example User interface
interface User {
    id: number;
    name: string;
    email: string;
}

// Example UserList component (for demonstration)
const UserList = () => {
    const { data, isLoading, error } = useQuery({
        queryKey: ['users'],
        queryFn: async () => {
            const response = await fetch('/api/users');
            if (!response.ok) throw new Error('Failed to fetch users');
            return response.json() as Promise<User[]>;
        },
    });

    if (isLoading) return <div>Loading users...</div>;
    if (error) return <div role="alert">Error: {(error as Error).message}</div>;

    return (
        <ul aria-label="User list">
            {data?.map((user) => (
                <li key={user.id}>
                    {user.name} - {user.email}
                </li>
            ))}
        </ul>
    );
};

describe('UserList Component', () => {
    beforeEach(() => {
        // Reset fetch mock before each test
        vi.resetAllMocks();
    });

    it('displays loading state initially', () => {
        // Mock fetch to never resolve
        global.fetch = vi.fn(() => new Promise(() => {})) as typeof fetch;

        renderWithProviders(<UserList />);

        expect(screen.getByText(/loading users/i)).toBeInTheDocument();
    });

    it('displays users after successful fetch', async () => {
        const mockUsers: User[] = [
            { id: 1, name: 'John Doe', email: 'john@example.com' },
            { id: 2, name: 'Jane Smith', email: 'jane@example.com' },
        ];

        global.fetch = vi.fn(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.resolve(mockUsers),
            } as Response)
        );

        renderWithProviders(<UserList />);

        await waitFor(() => {
            expect(screen.getByText(/john doe/i)).toBeInTheDocument();
        });

        expect(screen.getByText(/jane smith/i)).toBeInTheDocument();
        expect(screen.getByRole('list', { name: /user list/i })).toBeInTheDocument();
    });

    it('displays error message on fetch failure', async () => {
        global.fetch = vi.fn(() =>
            Promise.resolve({
                ok: false,
                status: 500,
            } as Response)
        );

        renderWithProviders(<UserList />);

        await waitFor(() => {
            expect(screen.getByRole('alert')).toBeInTheDocument();
        });

        expect(screen.getByText(/error.*failed to fetch users/i)).toBeInTheDocument();
    });

    it('handles network errors gracefully', async () => {
        global.fetch = vi.fn(() => Promise.reject(new Error('Network error')));

        renderWithProviders(<UserList />);

        await waitFor(() => {
            expect(screen.getByRole('alert')).toBeInTheDocument();
        });
    });

    it('displays empty list when no users exist', async () => {
        global.fetch = vi.fn(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.resolve([]),
            } as Response)
        );

        renderWithProviders(<UserList />);

        await waitFor(() => {
            const list = screen.getByRole('list', { name: /user list/i });
            expect(list.children).toHaveLength(0);
        });
    });
});
