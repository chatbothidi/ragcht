import pytest

from src.config import Settings


@pytest.fixture
def settings():
    return Settings(
        gcp_project_id="test-project",
        gcp_location="asia-northeast3",
        redis_url=None,
    )
