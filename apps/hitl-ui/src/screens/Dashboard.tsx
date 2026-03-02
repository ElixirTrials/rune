import { Button } from '../components/ui/Button';
import { DashboardCard } from '../features/dashboard/DashboardCard';
import { useHealthCheck } from '../hooks/useApi';
import { useAppStore } from '../stores/useAppStore';

export default function Dashboard() {
    const { data: health, isLoading, error } = useHealthCheck();
    const { sidebarOpen, toggleSidebar } = useAppStore();

    return (
        <div className="container mx-auto p-6">
            <header className="mb-8">
                <div className="flex items-center justify-between">
                    <h1 className="text-3xl font-bold text-foreground">
                        Human in the Loop Dashboard
                    </h1>
                    <Button onClick={toggleSidebar} variant="outline">
                        {sidebarOpen ? 'Close Sidebar' : 'Open Sidebar'}
                    </Button>
                </div>
                <p className="mt-2 text-muted-foreground">
                    Review and approve AI-generated content
                </p>
            </header>

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                <DashboardCard
                    title="System Status"
                    description="Current health of the backend services"
                >
                    {isLoading && <p className="text-muted-foreground">Checking health...</p>}
                    {error && (
                        <p className="text-destructive">Error: Unable to connect to backend</p>
                    )}
                    {health && (
                        <div className="flex items-center gap-2">
                            <span
                                className={`h-3 w-3 rounded-full ${
                                    health.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                                }`}
                            />
                            <span className="font-medium capitalize">{health.status}</span>
                        </div>
                    )}
                </DashboardCard>

                <DashboardCard title="Pending Reviews" description="Items awaiting human approval">
                    <p className="text-2xl font-bold">0</p>
                    <p className="text-sm text-muted-foreground">No pending items</p>
                </DashboardCard>

                <DashboardCard
                    title="Recent Activity"
                    description="Latest approvals and rejections"
                >
                    <p className="text-muted-foreground">No recent activity</p>
                </DashboardCard>
            </div>
        </div>
    );
}
