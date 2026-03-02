import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E Testing Configuration
 *
 * This configuration follows best practices for E2E testing:
 * - Multiple browser testing (Chromium, Firefox, WebKit)
 * - Mobile viewport testing
 * - Retry on failure in CI
 * - Video and screenshot capture on failure
 * - Parallel test execution
 */

export default defineConfig({
    // Directory containing E2E tests
    testDir: './e2e',

    // Maximum time one test can run
    timeout: 30 * 1000,

    // Test configuration
    expect: {
        // Maximum time expect() should wait for the condition to be met
        timeout: 5000,
    },

    // Run tests in files in parallel
    fullyParallel: true,

    // Fail the build on CI if you accidentally left test.only in the source code
    forbidOnly: !!process.env.CI,

    // Retry on CI only
    retries: process.env.CI ? 2 : 0,

    // Opt out of parallel tests on CI
    workers: process.env.CI ? 1 : undefined,

    // Reporter to use
    reporter: [
        ['html', { outputFolder: 'playwright-report' }],
        ['json', { outputFile: 'test-results/results.json' }],
        ['list'],
    ],

    // Shared settings for all the projects below
    use: {
        // Base URL to use in actions like `await page.goto('/')`
        baseURL: process.env.VITE_API_BASE_URL || 'http://localhost:3000',

        // Collect trace when retrying the failed test
        trace: 'on-first-retry',

        // Take screenshot on failure
        screenshot: 'only-on-failure',

        // Record video on failure
        video: 'retain-on-failure',

        // Maximum time each action (click, fill, etc.) can take
        actionTimeout: 10 * 1000,

        // Navigation timeout
        navigationTimeout: 30 * 1000,
    },

    // Configure projects for major browsers
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },

        {
            name: 'firefox',
            use: { ...devices['Desktop Firefox'] },
        },

        {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
        },

        // Mobile viewports
        {
            name: 'Mobile Chrome',
            use: { ...devices['Pixel 5'] },
        },
        {
            name: 'Mobile Safari',
            use: { ...devices['iPhone 12'] },
        },

        // Tablet
        {
            name: 'iPad',
            use: { ...devices['iPad Pro'] },
        },
    ],

    // Run your local dev server before starting the tests
    webServer: {
        command: 'npm run dev',
        url: 'http://localhost:3000',
        reuseExistingServer: !process.env.CI,
        timeout: 120 * 1000,
    },

    // Output directory for test artifacts
    outputDir: 'test-results/',
});
