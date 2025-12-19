from cache_unit_tests import LLMCachingUnitTests
from dheera_ai.caching import DheeraAICacheType

class TestGCSCacheUnitTests(LLMCachingUnitTests):
    def get_cache_type(self) -> DheeraAICacheType:
        return DheeraAICacheType.GCS
