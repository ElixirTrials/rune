/**
 * Example E2E Test: Form Submission with API Mocking
 *
 * This example demonstrates:
 * - Form filling and submission
 * - API route mocking
 * - Waiting for async operations
 * - Error handling in E2E tests
 */

import { expect, test } from '@playwright/test';

test.describe('Form Submission', () => {
    test('should submit a form successfully', async ({ page }) => {
        // Mock the API endpoint
        await page.route('**/api/submit', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    success: true,
                    message: 'Form submitted successfully',
                }),
            });
        });

        // Navigate to the form page
        await page.goto('/form');

        // Fill out the form
        await page.getByLabel(/name/i).fill('John Doe');
        await page.getByLabel(/email/i).fill('john@example.com');
        await page.getByLabel(/message/i).fill('This is a test message');

        // Submit the form
        await page.getByRole('button', { name: /submit/i }).click();

        // Wait for success message
        await expect(page.getByText(/form submitted successfully/i)).toBeVisible();
    });

    test('should handle form validation errors', async ({ page }) => {
        await page.goto('/form');

        // Submit empty form
        await page.getByRole('button', { name: /submit/i }).click();

        // Check for validation errors
        await expect(page.getByText(/name is required/i)).toBeVisible();
        await expect(page.getByText(/email is required/i)).toBeVisible();
    });

    test('should handle API errors gracefully', async ({ page }) => {
        // Mock API to return error
        await page.route('**/api/submit', async (route) => {
            await route.fulfill({
                status: 500,
                contentType: 'application/json',
                body: JSON.stringify({
                    error: 'Internal server error',
                }),
            });
        });

        await page.goto('/form');

        // Fill and submit form
        await page.getByLabel(/name/i).fill('John Doe');
        await page.getByLabel(/email/i).fill('john@example.com');
        await page.getByRole('button', { name: /submit/i }).click();

        // Check error message
        await expect(page.getByText(/error.*occurred/i)).toBeVisible();
    });

    test('should disable submit button during submission', async ({ page }) => {
        // Mock slow API response
        await page.route('**/api/submit', async (route) => {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ success: true }),
            });
        });

        await page.goto('/form');

        await page.getByLabel(/name/i).fill('John Doe');
        await page.getByLabel(/email/i).fill('john@example.com');

        const submitButton = page.getByRole('button', { name: /submit/i });
        await submitButton.click();

        // Button should be disabled during submission
        await expect(submitButton).toBeDisabled();

        // Wait for submission to complete
        await page.waitForResponse('**/api/submit');

        // Button should be enabled again
        await expect(submitButton).toBeEnabled();
    });
});

test.describe('Search Functionality', () => {
    test('should search and display results', async ({ page }) => {
        // Mock search API
        await page.route('**/api/search?q=*', async (route) => {
            const url = new URL(route.request().url());
            const query = url.searchParams.get('q');

            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    results: [
                        { id: 1, title: `Result for ${query}` },
                        { id: 2, title: `Another result for ${query}` },
                    ],
                }),
            });
        });

        await page.goto('/');

        // Type in search box
        const searchInput = page.getByPlaceholder(/search/i);
        await searchInput.fill('test query');

        // Wait for search results
        await expect(page.getByText(/result for test query/i)).toBeVisible();
        await expect(page.getByText(/another result for test query/i)).toBeVisible();
    });

    test('should debounce search requests', async ({ page }) => {
        let requestCount = 0;

        await page.route('**/api/search?q=*', async (route) => {
            requestCount++;
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ results: [] }),
            });
        });

        await page.goto('/');

        const searchInput = page.getByPlaceholder(/search/i);

        // Type quickly
        await searchInput.fill('a');
        await searchInput.fill('ab');
        await searchInput.fill('abc');

        // Wait for debounce
        await page.waitForTimeout(600);

        // Should have made only one request after debounce
        expect(requestCount).toBeLessThanOrEqual(1);
    });
});
