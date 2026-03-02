/**
 * Example Test: Custom Hook Testing
 *
 * This example demonstrates:
 * - Testing custom React hooks
 * - Using renderHook from @testing-library/react
 * - Testing hook state updates
 * - Testing hook side effects
 */

import { act, renderHook, waitFor } from '@testing-library/react';
import { useEffect, useState } from 'react';
import { describe, expect, it } from 'vitest';

// Example custom hook: useCounter
interface UseCounterOptions {
    initialValue?: number;
    min?: number;
    max?: number;
}

function useCounter({ initialValue = 0, min, max }: UseCounterOptions = {}) {
    const [count, setCount] = useState(initialValue);

    const increment = () => {
        setCount((prev) => {
            const next = prev + 1;
            return max !== undefined && next > max ? prev : next;
        });
    };

    const decrement = () => {
        setCount((prev) => {
            const next = prev - 1;
            return min !== undefined && next < min ? prev : next;
        });
    };

    const reset = () => {
        setCount(initialValue);
    };

    return { count, increment, decrement, reset };
}

describe('useCounter Hook', () => {
    it('initializes with default value', () => {
        const { result } = renderHook(() => useCounter());

        expect(result.current.count).toBe(0);
    });

    it('initializes with custom value', () => {
        const { result } = renderHook(() => useCounter({ initialValue: 10 }));

        expect(result.current.count).toBe(10);
    });

    it('increments count', () => {
        const { result } = renderHook(() => useCounter());

        act(() => {
            result.current.increment();
        });

        expect(result.current.count).toBe(1);

        act(() => {
            result.current.increment();
        });

        expect(result.current.count).toBe(2);
    });

    it('decrements count', () => {
        const { result } = renderHook(() => useCounter({ initialValue: 5 }));

        act(() => {
            result.current.decrement();
        });

        expect(result.current.count).toBe(4);
    });

    it('respects maximum value', () => {
        const { result } = renderHook(() => useCounter({ initialValue: 9, max: 10 }));

        act(() => {
            result.current.increment();
        });

        expect(result.current.count).toBe(10);

        // Try to increment beyond max
        act(() => {
            result.current.increment();
        });

        // Should stay at max
        expect(result.current.count).toBe(10);
    });

    it('respects minimum value', () => {
        const { result } = renderHook(() => useCounter({ initialValue: 1, min: 0 }));

        act(() => {
            result.current.decrement();
        });

        expect(result.current.count).toBe(0);

        // Try to decrement below min
        act(() => {
            result.current.decrement();
        });

        // Should stay at min
        expect(result.current.count).toBe(0);
    });

    it('resets to initial value', () => {
        const { result } = renderHook(() => useCounter({ initialValue: 5 }));

        act(() => {
            result.current.increment();
            result.current.increment();
        });

        expect(result.current.count).toBe(7);

        act(() => {
            result.current.reset();
        });

        expect(result.current.count).toBe(5);
    });
});

// Example: Testing a hook with side effects
function useLocalStorage(key: string, initialValue: string) {
    const [value, setValue] = useState(() => {
        const stored = localStorage.getItem(key);
        return stored !== null ? stored : initialValue;
    });

    useEffect(() => {
        localStorage.setItem(key, value);
    }, [key, value]);

    return [value, setValue] as const;
}

describe('useLocalStorage Hook', () => {
    it('initializes with value from localStorage', () => {
        localStorage.setItem('testKey', 'stored value');

        const { result } = renderHook(() => useLocalStorage('testKey', 'default'));

        expect(result.current[0]).toBe('stored value');
    });

    it('uses initial value when localStorage is empty', () => {
        localStorage.removeItem('testKey');

        const { result } = renderHook(() => useLocalStorage('testKey', 'default'));

        expect(result.current[0]).toBe('default');
    });

    it('updates localStorage when value changes', async () => {
        const { result } = renderHook(() => useLocalStorage('testKey', 'initial'));

        act(() => {
            result.current[1]('updated');
        });

        await waitFor(() => {
            expect(localStorage.getItem('testKey')).toBe('updated');
        });
    });
});
