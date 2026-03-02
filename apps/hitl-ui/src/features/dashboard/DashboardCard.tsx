import type { ReactNode } from 'react';

import { cn } from '../../lib/utils';

interface DashboardCardProps {
    title: string;
    description?: string;
    children: ReactNode;
    className?: string;
}

export function DashboardCard({ title, description, children, className }: DashboardCardProps) {
    return (
        <div
            className={cn(
                'rounded-lg border bg-card p-6 text-card-foreground shadow-sm',
                className
            )}
        >
            <div className="mb-4">
                <h3 className="text-lg font-semibold">{title}</h3>
                {description && <p className="text-sm text-muted-foreground">{description}</p>}
            </div>
            <div>{children}</div>
        </div>
    );
}
