import { Route, Routes } from 'react-router-dom';

import Dashboard from './screens/Dashboard';

function App() {
    return (
        <div className="min-h-screen bg-background">
            <Routes>
                <Route path="/" element={<Dashboard />} />
            </Routes>
        </div>
    );
}

export default App;
