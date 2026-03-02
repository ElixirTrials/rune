"""Example Unit Tests.

This module demonstrates best practices for unit testing in Python:
- Testing pure functions
- Using fixtures
- Parametrized tests
- Mocking dependencies
- Testing exceptions
- Testing async functions
"""

from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest


# Example function to test
def calculate_total_price(items: List[dict], tax_rate: float = 0.1) -> float:
    """Calculate total price including tax."""
    if not items:
        raise ValueError("Items list cannot be empty")

    subtotal = sum(item["price"] * item["quantity"] for item in items)
    total = subtotal * (1 + tax_rate)
    return round(total, 2)


class TestCalculateTotalPrice:
    """Test suite for calculate_total_price function."""

    def test_calculate_single_item(self):
        """Test calculation with a single item."""
        items = [{"price": 10.0, "quantity": 2}]
        result = calculate_total_price(items)

        assert result == 22.0  # (10 * 2) * 1.1

    def test_calculate_multiple_items(self):
        """Test calculation with multiple items."""
        items = [
            {"price": 10.0, "quantity": 2},
            {"price": 5.0, "quantity": 3},
        ]
        result = calculate_total_price(items)

        assert result == 38.5  # (20 + 15) * 1.1

    def test_custom_tax_rate(self):
        """Test calculation with custom tax rate."""
        items = [{"price": 100.0, "quantity": 1}]
        result = calculate_total_price(items, tax_rate=0.2)

        assert result == 120.0  # 100 * 1.2

    def test_zero_tax_rate(self):
        """Test calculation with zero tax."""
        items = [{"price": 50.0, "quantity": 2}]
        result = calculate_total_price(items, tax_rate=0)

        assert result == 100.0

    def test_empty_items_raises_error(self):
        """Test that empty items list raises ValueError."""
        with pytest.raises(ValueError, match="Items list cannot be empty"):
            calculate_total_price([])

    @pytest.mark.parametrize(
        "items,tax_rate,expected",
        [
            ([{"price": 10, "quantity": 1}], 0.1, 11.0),
            ([{"price": 20, "quantity": 2}], 0.15, 46.0),
            ([{"price": 100, "quantity": 1}], 0.0, 100.0),
        ],
    )
    def test_parametrized_calculations(self, items, tax_rate, expected):
        """Test multiple scenarios using parametrize."""
        result = calculate_total_price(items, tax_rate)
        assert result == expected


# Example class to test
class InventoryService:
    """Service for managing inventory."""

    def __init__(self, db_session):
        """Initialize the inventory service with a database session."""
        self.db = db_session

    def get_item_quantity(self, item_id: int) -> int:
        """Get quantity of an item."""
        # Use parameterized queries to prevent SQL injection
        result = self.db.query_one(
            "SELECT quantity FROM items WHERE id = ?", (item_id,)
        )
        return result["quantity"]

    def update_quantity(self, item_id: int, quantity: int) -> bool:
        """Update item quantity."""
        if quantity < 0:
            raise ValueError("Quantity cannot be negative")
        # Use parameterized queries to prevent SQL injection
        self.db.execute(
            "UPDATE items SET quantity = ? WHERE id = ?", (quantity, item_id)
        )
        return True


class TestInventoryService:
    """Test suite for InventoryService."""

    def test_get_item_quantity(self):
        """Test retrieving item quantity."""
        # Mock the database session
        mock_db = Mock()
        mock_db.query_one.return_value = {"quantity": 42}

        service = InventoryService(mock_db)
        result = service.get_item_quantity(item_id=1)

        assert result == 42
        mock_db.query_one.assert_called_once()

    def test_update_quantity_success(self):
        """Test successful quantity update."""
        mock_db = Mock()
        service = InventoryService(mock_db)

        result = service.update_quantity(item_id=1, quantity=10)

        assert result is True
        mock_db.execute.assert_called_once()

    def test_update_quantity_negative_raises_error(self):
        """Test that negative quantity raises ValueError."""
        mock_db = Mock()
        service = InventoryService(mock_db)

        with pytest.raises(ValueError, match="Quantity cannot be negative"):
            service.update_quantity(item_id=1, quantity=-5)

        # Verify database was not called
        mock_db.execute.assert_not_called()


# Example async function to test
async def fetch_user_data(user_id: int, api_client) -> dict:
    """Fetch user data from API."""
    response = await api_client.get(f"/users/{user_id}")
    if response.status_code == 404:
        raise ValueError(f"User {user_id} not found")
    return response.json()


class TestAsyncFunctions:
    """Test suite for async functions."""

    @pytest.mark.asyncio
    async def test_fetch_user_data_success(self):
        """Test successful user data fetch."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "John Doe"}
        mock_client.get.return_value = mock_response

        result = await fetch_user_data(1, mock_client)

        assert result == {"id": 1, "name": "John Doe"}
        mock_client.get.assert_awaited_once_with("/users/1")

    @pytest.mark.asyncio
    async def test_fetch_user_data_not_found(self):
        """Test user not found scenario."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_client.get.return_value = mock_response

        with pytest.raises(ValueError, match="User 1 not found"):
            await fetch_user_data(1, mock_client)


# Example: Testing with context manager
class TestWithPatch:
    """Examples using patch decorator and context manager."""

    @patch("time.time")
    def test_with_patch_decorator(self, mock_time):
        """Test using patch as decorator."""
        mock_time.return_value = 1234567890.0

        import time

        result = time.time()

        assert result == 1234567890.0

    def test_with_patch_context_manager(self):
        """Test using patch as context manager."""
        with patch("time.time", return_value=1234567890.0):
            import time

            result = time.time()
            assert result == 1234567890.0


# Example: Testing with custom fixtures
def test_with_sample_data(sample_user_data):
    """Test using fixture from conftest.py."""
    assert sample_user_data["username"] == "testuser"
    assert sample_user_data["email"] == "test@example.com"
