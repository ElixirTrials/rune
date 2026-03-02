/**
 * Example E2E Test: Basic Navigation and Interaction
 *
 * This example demonstrates:
 * - Page navigation
 * - Element interactions
 * - Assertions
 * - Accessibility testing with axe
 */

import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test.describe('Home Page', () => {
    test.beforeEach(async ({ page }) => {
        // Navigate to the home page before each test
        await page.goto('/');
    });

    test('should display the home page correctly', async ({ page }) => {
        // Check page title
        await expect(page).toHaveTitle(/ElixirTrials /i);

        // Check if main content is visible
        const main = page.locator('main');
        await expect(main).toBeVisible();
    });

    test('should navigate through the application', async ({ page }) => {
        // Click on a navigation link (adjust selector based on your app)
        await page.getByRole('link', { name: /about/i }).click();

        // Wait for navigation
        await page.waitForURL(/.*about.*/);

        // Verify we're on the about page
        await expect(page).toHaveURL(/.*about.*/);
    });

    test('should handle user interactions', async ({ page }) => {
        // Find and click a button (adjust based on your app)
        const button = page.getByRole('button', { name: /click me/i });
        await button.click();

        // Verify the result of the interaction
        await expect(page.getByText(/success/i)).toBeVisible();
    });

    test('should be accessible', async ({ page }) => {
        // Run accessibility scan
        const accessibilityScanResults = await new AxeBuilder({ page }).analyze();

        // Assert no accessibility violations
        expect(accessibilityScanResults.violations).toEqual([]);
    });

    test('should handle responsive design', async ({ page, viewport }) => {
        // This test runs on different viewports defined in playwright.config.ts

        // Check if the page adapts to the viewport
        const main = page.locator('main');
        await expect(main).toBeVisible();

        // Take a screenshot for visual comparison (optional)
        await page.screenshot({
            path: `test-results/home-${viewport?.width}x${viewport?.height}.png`,
        });
    });
});
