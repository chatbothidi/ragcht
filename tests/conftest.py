import pytest
import pytest_asyncio
from fakeredis import aioredis as fake_aioredis

from src.config import Settings


@pytest.fixture
def settings():
    return Settings(
        gcp_project_id="test-project",
        gcp_location="asia-northeast3",
        redis_url="redis://fake:6379/0",
    )


@pytest_asyncio.fixture
async def fake_redis():
    client = fake_aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()
