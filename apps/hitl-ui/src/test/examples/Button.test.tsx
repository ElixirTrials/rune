/**
 * Example Test: Simple Component Testing
 *
 * This example demonstrates:
 * - Basic component rendering
 * - Testing props
 * - Testing user interactions
 * - Testing accessibility
 */

import { describe, expect, it, vi } from 'vitest';
import { screen, userEvent } from '../utils';
import { renderWithProviders } from '../utils';

// Example Button component (for demonstration)
interface ButtonProps {
    children: React.ReactNode;
    onClick?: () => void;
    disabled?: boolean;
    variant?: 'primary' | 'secondary';
}

const Button = ({ children, onClick, disabled, variant = 'primary' }: ButtonProps) => {
    return (
        <button
            type="button"
            onClick={onClick}
            disabled={disabled}
            className={`btn btn-${variant}`}
            aria-label={typeof children === 'string' ? children : undefined}
        >
            {children}
        </button>
    );
};

describe('Button Component', () => {
    it('renders with correct text', () => {
        renderWithProviders(<Button>Click me</Button>);

        expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument();
    });

    it('calls onClick handler when clicked', async () => {
        const handleClick = vi.fn();
        const user = userEvent.setup();

        renderWithProviders(<Button onClick={handleClick}>Click me</Button>);

        const button = screen.getByRole('button');
        await user.click(button);

        expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('does not call onClick when disabled', async () => {
        const handleClick = vi.fn();
        const user = userEvent.setup();

        renderWithProviders(
            <Button onClick={handleClick} disabled>
                Click me
            </Button>
        );

        const button = screen.getByRole('button');
        expect(button).toBeDisabled();

        await user.click(button);

        expect(handleClick).not.toHaveBeenCalled();
    });

    it('applies correct variant class', () => {
        const { rerender } = renderWithProviders(<Button variant="primary">Primary</Button>);

        let button = screen.getByRole('button');
        expect(button).toHaveClass('btn-primary');

        rerender(<Button variant="secondary">Secondary</Button>);

        button = screen.getByRole('button');
        expect(button).toHaveClass('btn-secondary');
    });

    it('is accessible', () => {
        renderWithProviders(<Button>Accessible Button</Button>);

        const button = screen.getByRole('button', { name: /accessible button/i });
        expect(button).toBeInTheDocument();
        expect(button).toHaveAccessibleName();
    });
});
