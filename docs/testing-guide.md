# Testing Guide

This guide covers all testing frameworks and best practices for the ElixirTrials  project.

## Table of Contents

1. [Overview](#overview)
2. [Python Testing with pytest](#python-testing-with-pytest)
3. [React Testing with Vitest](#react-testing-with-vitest)
4. [E2E Testing with Playwright](#e2e-testing-with-playwright)
5. [Linting with ESLint](#linting-with-eslint)
6. [Best Practices](#best-practices)
7. [CI/CD Integration](#cicd-integration)

---

## Overview

This project uses modern testing frameworks:

- **Python Backend**: pytest + pytest-asyncio + pytest-cov
- **React Frontend**: Vitest + React Testing Library + Playwright
- **Linting**: ESLint (Frontend) + Ruff (Backend)

### Quick Start

```bash
# Python tests
make test-python

# React unit tests
cd apps/hitl-ui
npm test

# React E2E tests
cd apps/hitl-ui
npm run test:e2e

# Linting
make lint
```

---

## Python Testing with pytest

### Setup

pytest is already configured in `pyproject.toml`. All Python dependencies include testing tools:

```bash
# Install dependencies (includes test dependencies)
uv sync --dev
```

### Running Tests

```bash
# Run all Python tests
pytest

# Run tests for a specific component
pytest `services/api-service/tests/`

# Run tests with coverage
pytest --cov

# Run only unit tests (fast)
pytest -m unit

# Run integration tests
pytest -m integration

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest `services/api-service/tests/test_example_unit.py`

# Run specific test
pytest `services/api-service/tests/test_example_unit.py::TestCalculateTotalPrice::test_calculate_single_item`

# Run with verbose output
pytest -v

# Run and stop at first failure
pytest -x
```

### Test Structure

```
services/api-service/
├── src/
│   └── api_service/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_example_unit.py  # Unit tests
│   ├── test_example_integration.py  # Integration tests
│   └── test_example_mocking.py      # Mocking examples
└── pytest.ini                 # Component-specific config
```

### Writing Tests

#### Unit Test Example

```python
import pytest

def test_simple_function():
    """Test a simple function."""
    result = my_function(1, 2)
    assert result == 3

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_with_parameters(input, expected):
    """Test with multiple parameters."""
    assert double(input) == expected
```

#### Using Fixtures

```python
def test_with_fixture(db_session):
    """Test using a fixture from conftest.py."""
    # db_session is automatically provided
    user = User(name="Test")
    db_session.add(user)
    db_session.commit()

    assert user.id is not None
```

#### Async Tests

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async functions."""
    result = await fetch_data()
    assert result is not None
```

#### Testing Exceptions

```python
def test_raises_exception():
    """Test that function raises expected exception."""
    with pytest.raises(ValueError, match="Invalid input"):
        validate_input("bad_value")
```

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_unit():
    """Fast unit test."""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test with external dependencies."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Test that takes a long time."""
    pass
```

Run specific markers:
```bash
pytest -m unit        # Run only unit tests
pytest -m "not slow"  # Skip slow tests
```

### Mocking

#### Mock External APIs

```python
def test_with_mock_api(requests_mock):
    """Mock HTTP requests."""
    requests_mock.get(
        "https://api.example.com/data",
        json={"status": "success"}
    )

    result = fetch_from_api()
    assert result["status"] == "success"
```

#### Mock Async HTTP

```python
@pytest.mark.asyncio
async def test_async_http(aioresponses):
    """Mock async HTTP calls."""
    aioresponses.get(
        "https://api.example.com/data",
        payload={"data": "value"}
    )

    result = await fetch_async()
    assert result["data"] == "value"
```

#### Mock Environment Variables

```python
def test_with_env_var(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("API_KEY", "test-key")

    config = load_config()
    assert config.api_key == "test-key"
```

### Coverage

```bash
# Run with coverage
pytest --cov

# Generate HTML coverage report
pytest --cov --cov-report=html

# View coverage report
open htmlcov/index.html
```

Coverage is configured in `pyproject.toml` with these thresholds:
- Lines: 85%
- Functions: 85%
- Branches: 80%

---

## React Testing with Vitest

### Setup

Vitest is configured in `apps/hitl-ui/vite.config.ts`.

```bash
cd apps/hitl-ui
npm install
```

### Running Tests

```bash
# Run all tests
npm test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage

# Run in watch mode
npm test -- --watch

# Run specific test file
npm test -- `src/test/examples/Button.test.tsx`

# Run tests matching pattern
npm test -- --grep "Button"
```

### Test Structure

```
apps/hitl-ui/
├── src/
│   ├── components/
│   │   └── Button.tsx
│   └── test/
│       ├── setup.ts              # Test setup and global mocks
│       ├── utils.tsx             # Test utilities and helpers
│       └── examples/
│           ├── Button.test.tsx   # Component tests
│           ├── UserList.test.tsx # Async/API tests
│           └── useCounter.test.tsx # Hook tests
├── e2e/                          # Playwright E2E tests
├── vite.config.ts                # Vitest configuration
└── package.json
```

### Writing Component Tests

#### Basic Component Test

```typescript
import { describe, it, expect } from 'vitest';
import { screen, renderWithProviders } from '../utils';
import { MyComponent } from '../../components/MyComponent';

describe('MyComponent', () => {
    it('renders correctly', () => {
        renderWithProviders(<MyComponent title="Test" />);

        expect(screen.getByText('Test')).toBeInTheDocument();
    });
});
```

#### Testing User Interactions

```typescript
import { userEvent } from '../utils';

it('handles button click', async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(<Button onClick={handleClick}>Click me</Button>);

    await user.click(screen.getByRole('button'));

    expect(handleClick).toHaveBeenCalledTimes(1);
});
```

#### Testing Async Components

```typescript
import { waitFor } from '../utils';

it('loads data from API', async () => {
    global.fetch = vi.fn(() =>
        Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ data: 'test' }),
        })
    );

    renderWithProviders(<DataComponent />);

    await waitFor(() => {
        expect(screen.getByText('test')).toBeInTheDocument();
    });
});
```

#### Testing Custom Hooks

```typescript
import { renderHook, act } from '@testing-library/react';

it('increments counter', () => {
    const { result } = renderHook(() => useCounter());

    act(() => {
        result.current.increment();
    });

    expect(result.current.count).toBe(1);
});
```

### Test Utilities

Use the custom `renderWithProviders` utility for components that need providers:

```typescript
import { renderWithProviders } from '../test/utils';

// Automatically wraps component with Router and React Query providers
renderWithProviders(<MyComponent />);

// With initial route
renderWithProviders(<MyComponent />, { initialRoute: '/dashboard' });
```

### Mocking

#### Mock API Calls

```typescript
beforeEach(() => {
    global.fetch = vi.fn();
});

it('mocks fetch', async () => {
    global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: 'mocked' }),
    });

    // Your test code
});
```

#### Mock Modules

```typescript
vi.mock('../api/client', () => ({
    fetchUser: vi.fn(() => Promise.resolve({ id: 1, name: 'Test' })),
}));
```

### Coverage

```bash
# Generate coverage report
npm run test:coverage

# View HTML report
open coverage/index.html
```

Coverage thresholds (configured in vite.config.ts):
- Lines: 85%
- Functions: 85%
- Branches: 80%

---

## E2E Testing with Playwright

### Setup

Playwright is configured in `apps/hitl-ui/playwright.config.ts`.

```bash
cd apps/hitl-ui

# Install Playwright browsers (first time only)
npx playwright install
```

### Running E2E Tests

```bash
# Run all E2E tests
npm run test:e2e

# Run with UI mode (recommended for development)
npm run test:e2e:ui

# Run specific test file
npx playwright test e2e/example.spec.ts

# Run tests in specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit

# Run in headed mode (see the browser)
npx playwright test --headed

# Debug tests
npx playwright test --debug
```

### Writing E2E Tests

#### Basic Navigation Test

```typescript
import { test, expect } from '@playwright/test';

test('navigates to about page', async ({ page }) => {
    await page.goto('/');

    await page.getByRole('link', { name: /about/i }).click();

    await expect(page).toHaveURL(/.*about/);
    await expect(page.getByRole('heading', { name: /about/i })).toBeVisible();
});
```

#### Form Interaction

```typescript
test('submits form', async ({ page }) => {
    await page.goto('/contact');

    await page.getByLabel(/name/i).fill('John Doe');
    await page.getByLabel(/email/i).fill('john@example.com');
    await page.getByRole('button', { name: /submit/i }).click();

    await expect(page.getByText(/thank you/i)).toBeVisible();
});
```

#### API Mocking in E2E

```typescript
test('mocks API response', async ({ page }) => {
    await page.route('**/api/users', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([
                { id: 1, name: 'User 1' },
                { id: 2, name: 'User 2' },
            ]),
        });
    });

    await page.goto('/users');

    await expect(page.getByText('User 1')).toBeVisible();
});
```

#### Accessibility Testing

```typescript
import AxeBuilder from '@axe-core/playwright';

test('page is accessible', async ({ page }) => {
    await page.goto('/');

    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
});
```

### Test Organization

```typescript
test.describe('User Management', () => {
    test.beforeEach(async ({ page }) => {
        // Setup before each test
        await page.goto('/users');
    });

    test('displays user list', async ({ page }) => {
        // Test code
    });

    test('creates new user', async ({ page }) => {
        // Test code
    });
});
```

### Debugging E2E Tests

```bash
# Debug mode - pauses execution, opens browser
npx playwright test --debug

# Specific test in debug mode
npx playwright test example.spec.ts:10 --debug

# Generate trace (for debugging failures)
npx playwright test --trace on

# View trace
npx playwright show-trace trace.zip
```

---

## Linting with ESLint

### Setup

ESLint is configured in `apps/hitl-ui/eslint.config.js`.

### Running ESLint

```bash
cd apps/hitl-ui

# Lint all files
npm run lint

# Lint and auto-fix issues
npm run lint:fix

# Check code formatting
npm run format:check

# Auto-format code
npm run format
```

### ESLint Configuration

The project uses:
- **@typescript-eslint** for TypeScript linting
- **eslint-plugin-react** for React best practices
- **eslint-plugin-react-hooks** for React Hooks rules
- **eslint-plugin-jsx-a11y** for accessibility checks
- **eslint-plugin-prettier** for code formatting

### Common ESLint Rules

```typescript
// ✅ Good
const MyComponent = () => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        // Side effect
    }, [count]); // Dependency array

    return <div>{count}</div>;
};

// ❌ Bad - missing dependency array
useEffect(() => {
    // Side effect
}); // Warning: exhaustive-deps
```

---

## Best Practices

### Test Organization

1. **One test file per source file**
   ```
   src/components/Button.tsx
   src/components/Button.test.tsx
   ```

2. **Group related tests with describe**
   ```typescript
   describe('Button', () => {
       describe('when disabled', () => {
           it('does not call onClick', () => {});
       });
   });
   ```

3. **Use clear test names**
   ```typescript
   // ✅ Good
   it('displays error message when email is invalid', () => {});

   // ❌ Bad
   it('test1', () => {});
   ```

### Test Quality

1. **Follow AAA Pattern** (Arrange, Act, Assert)
   ```typescript
   it('adds two numbers', () => {
       // Arrange
       const a = 1;
       const b = 2;

       // Act
       const result = add(a, b);

       // Assert
       expect(result).toBe(3);
   });
   ```

2. **Test behavior, not implementation**
   ```typescript
   // ✅ Good - tests behavior
   expect(screen.getByRole('button')).toBeEnabled();

   // ❌ Bad - tests implementation
   expect(component.state.isDisabled).toBe(false);
   ```

3. **Use data-testid sparingly**
   ```typescript
   // ✅ Prefer semantic queries
   screen.getByRole('button', { name: /submit/i });
   screen.getByLabelText(/email/i);

   // ❌ Only use data-testid when necessary
   screen.getByTestId('submit-button');
   ```

### Mocking Guidelines

1. **Mock at the boundary**
   - Mock external APIs, not internal functions
   - Mock the database layer, not business logic

2. **Keep mocks simple**
   ```typescript
   // ✅ Good
   vi.fn(() => Promise.resolve({ data: 'test' }));

   // ❌ Bad - too complex
   vi.fn((url) => {
       if (url.includes('users')) {
           return Promise.resolve({ users: [] });
       } else if (url.includes('posts')) {
           return Promise.resolve({ posts: [] });
       }
       // ...many more conditions
   });
   ```

3. **Reset mocks between tests**
   ```typescript
   beforeEach(() => {
       vi.resetAllMocks();
   });
   ```

### Performance

1. **Use appropriate test types**
   - Unit tests: Fast, isolated, many
   - Integration tests: Medium speed, fewer
   - E2E tests: Slow, expensive, critical paths only

2. **Run tests in parallel**
   ```bash
   pytest -n auto  # Python
   npm test        # React (Vitest runs in parallel by default)
   ```

3. **Skip slow tests in development**
   ```bash
   pytest -m "not slow"
   ```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: uv sync --dev
      - name: Run tests
        run: pytest --cov

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: |
          cd apps/hitl-ui
          npm ci
      - name: Run tests
        run: |
          cd apps/hitl-ui
          npm test -- --run
      - name: Run E2E tests
        run: |
          cd apps/hitl-ui
          npx playwright install --with-deps
          npm run test:e2e
```

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true

      - id: eslint
        name: eslint
        entry: bash -c 'cd apps/hitl-ui && npm run lint'
        language: system
        pass_filenames: false
        files: \.tsx?$
```

---

## Troubleshooting

### Common Issues

#### Python: Import errors in tests
```bash
# Make sure PYTHONPATH is set (handled by pytest config)
export PYTHONPATH="${PYTHONPATH}:${PWD}/services/api-service/src"
```

#### React: Module not found
```bash
# Clear node_modules and reinstall
cd apps/hitl-ui
rm -rf node_modules package-lock.json
npm install
```

#### Playwright: Browser not installed
```bash
npx playwright install
```

#### Tests hanging
- Check for missing `await` in async tests
- Ensure mocks are properly set up
- Check for infinite loops in test code

---

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Playwright Documentation](https://playwright.dev/)
- [ESLint Rules](https://eslint.org/docs/rules/)

---

## Questions?

If you have questions about testing:
1. Check the example tests in `services/api-service/tests/`
2. Check the React examples in `apps/hitl-ui/src/test/examples/`
3. Refer to the official documentation linked above
