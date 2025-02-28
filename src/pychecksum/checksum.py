import hashlib
import time
from pathlib import Path
from typing import Optional, Union, Any

from .hashalgorithm import HashAlgorithm


class Checksum:
    # Default block size for file reading
    DEFAULT_BLOCK_SIZE: int = 65536  # 64KB is generally efficient

    # Default hash algorithm
    DEFAULT_HASH_ALGORITHM: HashAlgorithm = HashAlgorithm.SHA256

    def __init__(
            self,
            path: Union[str, Path],
            block_size: Optional[int] = None,
            hash_algorithm: Union[str, HashAlgorithm] = DEFAULT_HASH_ALGORITHM,
            delay: Optional[float] = None
    ) -> None:
        """
        Initialize a Checksum object.

        Args:
            path: Path to file for checksum calculation
            block_size: Custom block size for reading files
            hash_algorithm: Hash algorithm to use (either HashAlgorithm enum or string)

        Raises:
            ValueError: If path is None or hash_algorithm is not supported
        """
        if path is None:
            raise ValueError("Path cannot be None")

        self._delay = delay
        # Convert string algorithm to enum if needed
        if isinstance(hash_algorithm, str):
            hash_algorithm = HashAlgorithm.from_string(hash_algorithm)

        # Validate hash algorithm
        if not HashAlgorithm.is_available(hash_algorithm):
            available = ", ".join(sorted([a.value for a in HashAlgorithm.get_available()]))
            raise ValueError(f"Hash algorithm '{hash_algorithm.value}' is not available. "
                             f"Available algorithms: {available}")

        self._block_size: int = block_size or self.DEFAULT_BLOCK_SIZE
        self._path: Path = Path(path).absolute()
        self._hash_algorithm: HashAlgorithm = hash_algorithm
        self._checksum: Optional[str] = None

        # Only compute checksum if file exists
        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")
        if not self._path.is_file():
            raise ValueError(f"Not a valid file: {self._path}")
            # Special case for empty files
        self._checksum = self._compute_file_hash(self._path, self._delay)



    @property
    def checksum(self) -> Optional[str]:
        """Get the checksum value."""
        return self._checksum

    @property
    def path(self) -> Path:
        """Get the file path."""
        return self._path

    @property
    def block_size(self) -> int:
        """Get the block size used for this instance."""
        return self._block_size

    @property
    def hash_algorithm(self) -> HashAlgorithm:
        """Get the hash algorithm used."""
        return self._hash_algorithm

    @staticmethod
    def hash_data(data: bytes, algorithm: Union[str, HashAlgorithm] = DEFAULT_HASH_ALGORITHM) -> str:
        """Compute hash of data in memory using the specified algorithm.

        Args:
            data: Bytes to hash
            algorithm: Hash algorithm to use (either HashAlgorithm enum or string)

        Returns:
            Hexadecimal hash digest

        Raises:
            ValueError: If algorithm is not supported
        """
        # Convert string algorithm to enum if needed
        if isinstance(algorithm, str):
            algorithm = HashAlgorithm.from_string(algorithm)

        # Validate hash algorithm
        if not HashAlgorithm.is_available(algorithm):
            available = ", ".join(sorted([a.value for a in HashAlgorithm.get_available()]))
            raise ValueError(f"Hash algorithm '{algorithm.value}' is not available. "
                             f"Available algorithms: {available}")

        return getattr(hashlib, algorithm.value)(data).hexdigest()

    def _compute_file_hash(self, file_path: Path, delay:Optional[float]=None) -> str:
        """Compute hash of a file using the current hash algorithm.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal hash digest
        """
        # Ensure we're using absolute path
        file_path = file_path.absolute()

        # Handle empty files quickly
        if file_path.stat().st_size == 0:
            return HashAlgorithm.get_empty_hash(self._hash_algorithm)

        # Create hash object for the selected algorithm
        h = getattr(hashlib, self._hash_algorithm.value)()

        with open(file_path, "rb") as f:
            # Read once with a generator to avoid multiple disk reads
            for block in iter(lambda: f.read(self._block_size), b""):
                h.update(block)
                if delay:
                    time.sleep(delay)

        return h.hexdigest()

    @classmethod
    def compute_hash(
            cls,
            file_path: Union[str, Path],
            hash_algorithm: Union[str, HashAlgorithm] = DEFAULT_HASH_ALGORITHM,
            block_size: Optional[int] = None,
            delay: Optional[float] = None
    ) -> str:
        """Compute hash of a file using the specified algorithm.

        Args:
            file_path: Path to file
            hash_algorithm: Hash algorithm to use (either HashAlgorithm enum or string)
            block_size: Custom block size for reading files

        Returns:
            Hexadecimal hash digest

        Raises:
            ValueError: If hash_algorithm is not supported
        """
        # Convert to Path and absolute path
        absolute_path = Path(file_path).absolute()

        # Create a temporary Checksum object with the specified algorithm
        checksum = cls(absolute_path, block_size, hash_algorithm, delay)

        # Return the computed checksum
        if checksum.checksum is None:
            raise ValueError(f"Failed to compute checksum for {file_path}")
        return checksum.checksum

    # Legacy methods for backward compatibility
    @classmethod
    def sha256(cls, file_path: Union[str, Path], block_size: Optional[int] = None) -> str:
        """Compute SHA-256 hash of a file (for backward compatibility).

        Args:
            file_path: Path to file
            block_size: Custom block size for reading files

        Returns:
            Hexadecimal hash digest
        """
        return cls.compute_hash(file_path, HashAlgorithm.SHA256, block_size)

    @staticmethod
    def sha256_data(data: bytes) -> str:
        """Compute SHA-256 hash of data (for backward compatibility).

        Args:
            data: Bytes to hash

        Returns:
            Hexadecimal hash digest
        """
        return Checksum.hash_data(data, HashAlgorithm.SHA256)

    def __eq__(self, other: Any) -> bool:
        """Enable direct comparison of Checksum objects."""
        if isinstance(other, Checksum):
            return (self.checksum == other.checksum and
                    self.hash_algorithm == other.hash_algorithm)
        return False

    def __str__(self) -> str:
        """String representation of the checksum."""
        return self.checksum or "No checksum computed"

    def __repr__(self) -> str:
        """Formal representation of the Checksum object."""
        return (f"Checksum(path='{self._path}', "
                f"block_size={self._block_size}, "
                f"hash_algorithm={self._hash_algorithm!r})")