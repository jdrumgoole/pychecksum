from pathlib import Path
from typing import Dict, Optional, NamedTuple, Union

from .hashalgorithm import HashAlgorithm


class HashCacheKey(NamedTuple):
    """A key for the hash cache, including the file path, modification time, size, and algorithm."""
    path: str
    mtime: float
    size: int
    algorithm: HashAlgorithm


class HashCache:
    """A cache manager for file hash operations."""

    def __init__(self, max_size: int = 128):
        """
        Initialize the hash cache.

        Args:
            max_size: Maximum number of entries to store in cache
        """
        self._cache: Dict[HashCacheKey, str] = {}
        self._max_size: int = max_size

    @staticmethod
    def _get_file_key(file_path: Path, algorithm: HashAlgorithm) -> HashCacheKey:
        """
        Generate a cache key for a file based on absolute path, mtime, size, and algorithm.

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm used

        Returns:
            A HashCacheKey that uniquely identifies the file state and algorithm
        """
        # Convert to absolute path to ensure consistency
        abs_path = file_path.absolute()
        file_stat = abs_path.stat()
        return HashCacheKey(
            path=str(abs_path),
            mtime=file_stat.st_mtime,
            size=file_stat.st_size,
            algorithm=algorithm
        )

    def get(self, file_path: Path, algorithm: Union[str, HashAlgorithm]) -> Optional[str]:
        """
        Get a cached hash value if available.

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm used (either HashAlgorithm enum or string)

        Returns:
            The cached hash or None if not in cache
        """
        try:
            # Convert string algorithm to enum if needed
            if isinstance(algorithm, str):
                algorithm = HashAlgorithm.from_string(algorithm)

            # Always use absolute path
            abs_path = file_path.absolute()
            key = self._get_file_key(abs_path, algorithm)
            return self._cache.get(key)
        except (FileNotFoundError, PermissionError, ValueError):
            return None

    def set(self, file_path: Path, hash_value: str, algorithm: Union[str, HashAlgorithm]) -> None:
        """
        Store a hash value in the cache.

        Args:
            file_path: Path to the file
            hash_value: The hash value to cache
            algorithm: Hash algorithm used (either HashAlgorithm enum or string)
        """
        try:
            # Convert string algorithm to enum if needed
            if isinstance(algorithm, str):
                algorithm = HashAlgorithm.from_string(algorithm)

            # Always use absolute path
            abs_path = file_path.absolute()
            key = self._get_file_key(abs_path, algorithm)

            # Manage cache size
            if len(self._cache) >= self._max_size:
                # Remove oldest entry (simple approach)
                self._cache.pop(next(iter(self._cache)))

            self._cache[key] = hash_value
        except (FileNotFoundError, PermissionError, ValueError):
            pass  # Don't cache if we can't get file stats or invalid algorithm

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def invalidate(
            self,
            file_path: Path,
            algorithm: Optional[Union[str, HashAlgorithm]] = None
    ) -> None:
        """
        Remove specific file entries from the cache.

        Args:
            file_path: Path to the file to remove from cache
            algorithm: If provided, only invalidate entries for this algorithm.
                       If None, invalidate all entries for the file regardless of algorithm.
        """
        try:
            # Always use absolute path
            abs_path = file_path.absolute()

            # Convert string algorithm to enum if needed
            if algorithm is not None and isinstance(algorithm, str):
                algorithm = HashAlgorithm.from_string(algorithm)

            # If algorithm is specified, only invalidate that specific entry
            if algorithm is not None:
                try:
                    key = self._get_file_key(abs_path, algorithm)
                    if key in self._cache:
                        del self._cache[key]
                except (FileNotFoundError, PermissionError):
                    pass
            else:
                # If no algorithm specified, invalidate all entries for this file
                # We need to make a list of keys to avoid modifying during iteration
                keys_to_remove = [
                    key for key in self._cache.keys()
                    if key.path == str(abs_path)
                ]
                for key in keys_to_remove:
                    del self._cache[key]

        except Exception:
            # Fail silently for any other errors
            pass