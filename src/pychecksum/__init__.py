"""
Checksum package for computing and caching file hashes using various algorithms.

This package provides:
- HashAlgorithm: Enum representing supported hash algorithms
- Checksum: Base class for computing file hashes with any hashlib algorithm
- HashCache: Class for caching hash values based on file modification time and size
- CachedChecksum: Wrapper class that adds caching to the Checksum class
"""

from .hashalgorithm import HashAlgorithm, EMPTY_HASHES
from .checksum import Checksum
from .hashcache import HashCache
from .cachedchecksum import CachedChecksum

__all__ = ['HashAlgorithm', 'Checksum', 'HashCache', 'CachedChecksum', 'EMPTY_HASHES']