from cache_unit_tests import LLMCachingUnitTests
from dheera_ai.caching import DheeraAICacheType


class TestDiskCacheUnitTests(LLMCachingUnitTests):
    def get_cache_type(self) -> DheeraAICacheType:
        return DheeraAICacheType.DISK


# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])
