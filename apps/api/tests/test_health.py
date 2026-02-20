"""Tests for health and root endpoints."""
import pytest


@pytest.mark.asyncio
async def test_read_root(client):
    """GET / returns Hello World."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data == {"Hello": "World"}


@pytest.mark.asyncio
async def test_health_check(client):
    """GET /health returns healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data
