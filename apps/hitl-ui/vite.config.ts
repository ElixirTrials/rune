import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';

export default defineConfig(({ mode }) => {
    void mode;
    const rawBasePath = process.env.BASE_PATH ?? '/demo-app/';
    const trimmedBasePath = rawBasePath.trim();
    const withLeadingSlash = trimmedBasePath
        ? trimmedBasePath.startsWith('/')
            ? trimmedBasePath
            : `/${trimmedBasePath}`
        : '/';
    const normalizedBasePath = withLeadingSlash.endsWith('/')
        ? withLeadingSlash
        : `${withLeadingSlash}/`;
    return {
        plugins: [react()],
        base: normalizedBasePath,
        resolve: {
            extensions: ['.js', '.jsx', '.ts', '.tsx', '.json'],
            alias: {
                '@': path.resolve(__dirname, './src'),
            },
        },
        build: {
            target: 'esnext',
            outDir: 'build',
            chunkSizeWarningLimit: 800,
        },
        server: {
            port: 3000,
            open: true,
        },
        test: {
            globals: true,
            environment: 'jsdom',
            setupFiles: './src/test/setup.ts',
            exclude: ['e2e/**', 'node_modules/**'],
            pool: 'threads',
            css: true,
            coverage: {
                provider: 'v8',
                reporter: ['text', 'json', 'html'],
                exclude: [
                    'node_modules/',
                    'src/test/',
                    '**/*.d.ts',
                    '**/*.config.*',
                    '**/mock*.ts',
                    '**/*.test.*',
                    '**/*.spec.*',
                ],
                thresholds: {
                    lines: 85,
                    functions: 85,
                    branches: 80,
                    statements: 85,
                },
            },
        },
    };
});
