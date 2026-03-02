import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
        },
    });

    if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
}

interface HealthResponse {
    status: string;
}

export function useHealthCheck() {
    return useQuery({
        queryKey: ['health'],
        queryFn: () => fetchApi<HealthResponse>('/health'),
        refetchInterval: 30000, // Check every 30 seconds
    });
}

interface ReadinessResponse {
    status: string;
    database: string;
}

export function useReadinessCheck() {
    return useQuery({
        queryKey: ['ready'],
        queryFn: () => fetchApi<ReadinessResponse>('/ready'),
    });
}

interface Task {
    id: string;
    status: string;
    input_data: Record<string, unknown>;
    output_data: Record<string, unknown>;
}

export function useTasks() {
    return useQuery({
        queryKey: ['tasks'],
        queryFn: () => fetchApi<Task[]>('/api/tasks'),
    });
}

export function useApproveTask() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (taskId: string) =>
            fetchApi(`/api/tasks/${taskId}/approve`, { method: 'POST' }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['tasks'] });
        },
    });
}

export function useRejectTask() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (taskId: string) => fetchApi(`/api/tasks/${taskId}/reject`, { method: 'POST' }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['tasks'] });
        },
    });
}
