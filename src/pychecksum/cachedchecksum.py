from pathlib import Path
from typing import Optional, Union, Any

from .checksum import Checksum
from .hashcache import HashCache
from .hashalgorithm import HashAlgorithm


class CachedChecksum:
    """A wrapper that adds caching to Checksum operations."""

    # Singleton cache object shared by all instances
    _cache = HashCache(max_size=128)

    def __init__(
            self,
            path: Union[str, Path],
            block_size: Optional[int] = None,
            hash_algorithm: Union[str, HashAlgorithm] = Checksum.DEFAULT_HASH_ALGORITHM,
            use_cache: bool = True
    ) -> None:
        """
        Initialize a CachedChecksum object.

        Args:
            path: Path to file for checksum calculation
            block_size: Custom block size for reading files
            hash_algorithm: Hash algorithm to use (either HashAlgorithm enum or string)
            use_cache: Whether to use cached results

        Raises:
            ValueError: If path is None or hash_algorithm is not supported
        """
        if path is None:
            raise ValueError("Path cannot be None")

        # Convert string algorithm to enum if needed
        if isinstance(hash_algorithm, str):
            hash_algorithm = HashAlgorithm.from_string(hash_algorithm)

        self._use_cache: bool = use_cache
        self._path: Path = Path(path).absolute()
        self._hash_algorithm: HashAlgorithm = hash_algorithm

        # The inner checksum object handles the actual hash computation
        if not use_cache:
            # If cache is disabled, just use a regular Checksum
            self._checksum = Checksum(path, block_size, hash_algorithm)
        else:
            # If caching is enabled, try to get from cache first
            if self._path.exists() and self._path.is_file():
                cached_hash = self._cache.get(self._path, self._hash_algorithm)

                if cached_hash:
                    # Create a Checksum with the cached hash
                    self._checksum = Checksum(self._path, block_size, hash_algorithm)
                    # Override the computed checksum with our cached value
                    self._checksum._checksum = cached_hash
                else:
                    # Compute and cache the hash
                    self._checksum = Checksum(path, block_size, hash_algorithm)
                    if self._checksum.checksum:
                        self._cache.set(self._path, self._checksum.checksum, self._hash_algorithm)
            else:
                # Path doesn't exist or is not a file
                self._checksum = Checksum(path, block_size, hash_algorithm)

    @property
    def checksum(self) -> Optional[str]:
        """Get the checksum value."""
        return self._checksum.checksum

    @property
    def path(self) -> Path:
        """Get the file path."""
        return self._checksum.path

    @property
    def block_size(self) -> int:
        """Get the block size used for this instance."""
        return self._checksum.block_size

    @property
    def hash_algorithm(self) -> HashAlgorithm:
        """Get the hash algorithm used."""
        return self._checksum.hash_algorithm

    @classmethod
    def hash_data(
            cls,
            data: bytes,
            algorithm: Union[str, HashAlgorithm] = Checksum.DEFAULT_HASH_ALGORITHM
    ) -> str:
        """Compute hash of data using the specified algorithm.

        Args:
            data: Bytes to hash
            algorithm: Hash algorithm to use (either HashAlgorithm enum or string)

        Returns:
            Hexadecimal hash digest
        """
        return Checksum.hash_data(data, algorithm)

    @classmethod
    def compute_hash(
            cls,
            file_path: Union[str, Path],
            hash_algorithm: Union[str, HashAlgorithm] = Checksum.DEFAULT_HASH_ALGORITHM,
            block_size: Optional[int] = None,
            use_cache: bool = True
    ) -> str:
        """
        Compute hash of a file with optional caching.

        Args:
            file_path: Path to file
            hash_algorithm: Hash algorithm to use (either HashAlgorithm enum or string)
            block_size: Custom block size for reading files
            use_cache: Whether to use cached results

        Returns:
            Hexadecimal hash digest
        """
        # Convert string algorithm to enum if needed
        if isinstance(hash_algorithm, str):
            hash_algorithm = HashAlgorithm.from_string(hash_algorithm)

        # Convert to absolute path
        abs_path = Path(file_path).absolute()

        if not use_cache:
            return Checksum.compute_hash(abs_path, hash_algorithm, block_size)

        cached_hash = cls._cache.get(abs_path, hash_algorithm)
        if cached_hash:
            return cached_hash

        # Compute and cache the hash
        hash_value = Checksum.compute_hash(abs_path, hash_algorithm, block_size)
        cls._cache.set(abs_path, hash_value, hash_algorithm)
        return hash_value

    # Legacy methods for backward compatibility
    @classmethod
    def sha256(cls, file_path: Union[str, Path], use_cache: bool = True) -> str:
        """
        Compute SHA-256 hash of a file with optional caching (for backward compatibility).

        Args:
            file_path: Path to file
            use_cache: Whether to use cached results

        Returns:
            Hexadecimal hash digest
        """
        return cls.compute_hash(file_path, HashAlgorithm.SHA256, use_cache=use_cache)

    @classmethod
    def sha256_data(cls, data: bytes) -> str:
        """
        Compute SHA-256 hash of data (for backward compatibility).

        Args:
            data: Bytes to hash

        Returns:
            Hexadecimal hash digest
        """
        return cls.hash_data(data, HashAlgorithm.SHA256)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the entire hash cache."""
        cls._cache.clear()

    @classmethod
    def invalidate_cache_entry(
            cls,
            file_path: Union[str, Path],
            algorithm: Optional[Union[str, HashAlgorithm]] = None
    ) -> None:
        """
        Remove specific file entries from the cache.

        Args:
            file_path: Path to the file to remove from cache
            algorithm: If provided, only invalidate entries for this algorithm.
                      If None, invalidate all entries for the file regardless of algorithm.
        """
        abs_path = Path(file_path).absolute()
        cls._cache.invalidate(abs_path, algorithm)

    def __eq__(self, other: Any) -> bool:
        """Enable direct comparison of Checksum objects."""
        if isinstance(other, (CachedChecksum, Checksum)):
            return (self.checksum == other.checksum and
                    self.hash_algorithm == other.hash_algorithm)
        return False

    def __str__(self) -> str:
        """String representation of the checksum."""
        return str(self._checksum)

    def __repr__(self) -> str:
        """Formal representation of the CachedChecksum object."""
        return (f"CachedChecksum(path='{self.path}', "
                f"block_size={self.block_size}, "
                f"hash_algorithm={self.hash_algorithm!r}, "
                f"use_cache={self._use_cache})")