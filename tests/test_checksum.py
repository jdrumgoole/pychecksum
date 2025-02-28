import os
import time
import pytest
import hashlib
import tempfile
from pathlib import Path

# Import all necessary classes from the package
from pychecksum import HashAlgorithm, Checksum, HashCache, CachedChecksum, EMPTY_HASHES


class TestHashAlgorithm:
    """Tests for the HashAlgorithm enum."""

    def test_enum_values(self):
        """Test that enum values match hashlib algorithm names."""
        for algo in HashAlgorithm:
            assert algo.value in hashlib.algorithms_available or algo.value in dir(hashlib)

    def test_from_string(self):
        """Test conversion from string to enum."""
        # Test valid algorithms
        assert HashAlgorithm.from_string('sha256') == HashAlgorithm.SHA256
        assert HashAlgorithm.from_string('SHA256') == HashAlgorithm.SHA256  # Case insensitive

        # Test invalid algorithm
        with pytest.raises(ValueError):
            HashAlgorithm.from_string('invalid_algorithm')

    def test_is_available(self):
        """Test checking if an algorithm is available."""
        # SHA256 should be available on all platforms
        assert HashAlgorithm.is_available(HashAlgorithm.SHA256)

        # Check actual availability of other algorithms
        for algo in HashAlgorithm:
            expected = algo.value in hashlib.algorithms_available
            assert HashAlgorithm.is_available(algo) == expected

    def test_get_available(self):
        """Test getting available algorithms."""
        available = HashAlgorithm.get_available()
        assert isinstance(available, set)
        assert all(isinstance(algo, HashAlgorithm) for algo in available)
        assert HashAlgorithm.SHA256 in available  # SHA256 should be available on all platforms

    def test_get_empty_hash(self):
        """Test getting empty hash for different algorithms."""
        # Test predefined empty hashes
        for algo, expected_hash in EMPTY_HASHES.items():
            assert HashAlgorithm.get_empty_hash(algo) == expected_hash

        # Test computing empty hash for an algorithm not in EMPTY_HASHES
        # This is a bit tricky to test directly, so we'll mock it
        if hasattr(hashlib, 'non_existent_algo'):
            # Just skip this test if by chance this algorithm exists
            pass
        else:
            # Create a test algorithm by monkey-patching hashlib temporarily
            original_algorithms = HashAlgorithm.get_available()
            test_hash = "test_empty_hash"

            class MockHash:
                def __init__(self, data=b''):
                    self.data = data

                def hexdigest(self):
                    return test_hash

            # Add a mock algorithm to hashlib temporarily
            setattr(hashlib, 'mock_algo', lambda data=b'': MockHash(data))

            # Add it to algorithms_available temporarily
            if not hasattr(hashlib, 'algorithms_available'):
                hashlib.algorithms_available = set()
            original_algos = hashlib.algorithms_available.copy()
            hashlib.algorithms_available.add('mock_algo')


class TestBaseChecksum:
    """Tests for the base Checksum class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield Path(tmpdirname)

    @pytest.fixture
    def empty_file(self, temp_dir):
        """Create an empty test file."""
        file_path = temp_dir / "empty.txt"
        file_path.touch()
        return file_path

    @pytest.fixture
    def small_file(self, temp_dir):
        """Create a small test file with known content."""
        file_path = temp_dir / "small.txt"
        with open(file_path, "wb") as f:
            f.write(b"Hello, World!")
        return file_path

    @pytest.fixture
    def large_file(self, temp_dir):
        """Create a large test file that exceeds the block size."""
        file_path = temp_dir / "large.txt"
        # Create a file larger than the default block size
        with open(file_path, "wb") as f:
            # Write 100KB of data (exceeds the default 64KB block size)
            f.write(b"X" * 102400)
        return file_path

    def test_init_with_none_path(self):
        """Test initialization with None as the file path raises ValueError."""
        with pytest.raises(ValueError, match="Path cannot be None"):
            Checksum(None)

    def test_init_with_invalid_algorithm(self, small_file):
        """Test initialization with an invalid algorithm raises ValueError."""
        with pytest.raises(ValueError):
            Checksum(small_file, hash_algorithm="invalid_algorithm")

    def test_init_with_string_algorithm(self, small_file):
        """Test initialization with a valid string algorithm."""
        checksum = Checksum(small_file, hash_algorithm="sha256")
        assert checksum.hash_algorithm == HashAlgorithm.SHA256

    def test_init_with_enum_algorithm(self, small_file):
        """Test initialization with an enum algorithm."""
        checksum = Checksum(small_file, hash_algorithm=HashAlgorithm.SHA256)
        assert checksum.hash_algorithm == HashAlgorithm.SHA256

    def test_empty_file_hash(self, empty_file):
        """Test hash calculation for an empty file."""
        for algo in HashAlgorithm.get_available():
            checksum = Checksum(empty_file, hash_algorithm=algo)
            expected_hash = HashAlgorithm.get_empty_hash(algo)
            assert checksum.checksum == expected_hash

    def test_small_file_hash(self, small_file):
        """Test hash calculation for a small file with multiple algorithms."""
        content = b"Hello, World!"

        for algo in HashAlgorithm.get_available():
            # Manually calculate the expected hash
            h = getattr(hashlib, algo.value)(content)
            expected_hash = h.hexdigest()

            # Test the Checksum class
            checksum = Checksum(small_file, hash_algorithm=algo)
            assert checksum.checksum == expected_hash

    def test_delay_hash(self, small_file):
        """Test hash calculation for a small file with multiple algorithms."""
        content = b"Hello, World!"

        for algo in HashAlgorithm.get_available():
            # Manually calculate the expected hash
            h = getattr(hashlib, algo.value)(content)
            expected_hash = h.hexdigest()

            # Test the Checksum class
            delay = 0.1
            start = time.time()
            checksum = Checksum(small_file, hash_algorithm=algo, delay=delay)
            end = time.time()
            duration = end - start
            assert duration >= delay
            assert checksum.checksum == expected_hash

    def test_large_file_hash(self, large_file):
        """Test hash calculation for a file larger than the block size."""
        # Test with a few key algorithms
        for algo in [HashAlgorithm.SHA256, HashAlgorithm.MD5]:
            if not HashAlgorithm.is_available(algo):
                continue

            # Manually calculate the expected hash
            with open(large_file, "rb") as f:
                data = f.read()
            h = getattr(hashlib, algo.value)(data)
            expected_hash = h.hexdigest()

            # Test the Checksum class
            checksum = Checksum(large_file, hash_algorithm=algo)
            assert checksum.checksum == expected_hash

    def test_custom_block_size(self, large_file):
        """Test using a custom block size."""
        custom_block_size = 8192  # 8KB

        for algo in [HashAlgorithm.SHA256, HashAlgorithm.MD5]:
            if not HashAlgorithm.is_available(algo):
                continue

            checksum = Checksum(large_file, block_size=custom_block_size, hash_algorithm=algo)
            assert checksum.block_size == custom_block_size

            # Manually calculate the expected hash
            with open(large_file, "rb") as f:
                data = f.read()
            h = getattr(hashlib, algo.value)(data)
            expected_hash = h.hexdigest()

            assert checksum.checksum == expected_hash

    def test_hash_data_method(self):
        """Test the hash_data static method with different algorithms."""
        test_data = b"Test data for hashing"

        for algo in HashAlgorithm.get_available():
            # Calculate expected hash
            h = getattr(hashlib, algo.value)(test_data)
            expected_hash = h.hexdigest()

            # Test string algorithm
            assert Checksum.hash_data(test_data, algo.value) == expected_hash

            # Test enum algorithm
            assert Checksum.hash_data(test_data, algo) == expected_hash

    def test_compute_hash_class_method(self, small_file):
        """Test the compute_hash class method with different algorithms."""
        for algo in HashAlgorithm.get_available():
            # Manually calculate the expected hash
            with open(small_file, "rb") as f:
                data = f.read()
            h = getattr(hashlib, algo.value)(data)
            expected_hash = h.hexdigest()

            # Test string algorithm
            assert Checksum.compute_hash(small_file, algo.value) == expected_hash

            # Test enum algorithm
            assert Checksum.compute_hash(small_file, algo) == expected_hash

    def test_legacy_sha256_method(self, small_file):
        """Test the legacy sha256 class method."""
        # Manually calculate the expected hash
        with open(small_file, "rb") as f:
            data = f.read()
        expected_hash = hashlib.sha256(data).hexdigest()

        # Test the class method
        assert Checksum.sha256(small_file) == expected_hash

    def test_legacy_sha256_data_method(self):
        """Test the legacy sha256_data static method."""
        test_data = b"Test data for SHA-256"
        expected_hash = hashlib.sha256(test_data).hexdigest()
        assert Checksum.sha256_data(test_data) == expected_hash

    def test_equality_comparison(self, small_file):
        """Test equality comparison between Checksum objects."""
        # Same file, same algorithm
        checksum1 = Checksum(small_file, hash_algorithm=HashAlgorithm.SHA256)
        checksum2 = Checksum(small_file, hash_algorithm=HashAlgorithm.SHA256)
        assert checksum1 == checksum2

        # Same file, different algorithms
        if HashAlgorithm.is_available(HashAlgorithm.MD5):
            checksum3 = Checksum(small_file, hash_algorithm=HashAlgorithm.MD5)
            assert checksum1 != checksum3

        # Test comparison with a non-Checksum object
        assert checksum1 != "not a checksum object"

    def test_string_representation(self, small_file):
        """Test string representation of Checksum objects."""
        checksum = Checksum(small_file)
        assert str(checksum) == checksum.checksum

    def test_repr_representation(self, small_file):
        """Test the repr representation of Checksum objects."""
        checksum = Checksum(small_file, hash_algorithm=HashAlgorithm.SHA256)
        assert repr(checksum).startswith("Checksum(path='")
        assert f"hash_algorithm={HashAlgorithm.SHA256!r}" in repr(checksum)


class TestHashCache:
    """Tests for the HashCache class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield Path(tmpdirname)

    @pytest.fixture
    def test_file(self, temp_dir):
        """Create a test file."""
        file_path = temp_dir / "test.txt"
        with open(file_path, "wb") as f:
            f.write(b"Test content")
        return file_path

    def test_cache_set_get(self, test_file):
        """Test setting and getting cache entries with different algorithms."""
        cache = HashCache()
        test_hash = "test_hash_value"

        for algo in HashAlgorithm.get_available():
            # Set a value in the cache with enum
            cache.set(test_file, test_hash, algo)

            # Get the value from the cache with enum
            cached_hash = cache.get(test_file, algo)
            assert cached_hash == test_hash

            # Get the value from the cache with string
            cached_hash = cache.get(test_file, algo.value)
            assert cached_hash == test_hash

    def test_cache_invalidate_specific_algorithm(self, test_file):
        """Test invalidating a specific algorithm cache entry."""
        cache = HashCache()
        test_hash = "test_hash_value"

        # Add entries for multiple algorithms
        for algo in list(HashAlgorithm.get_available())[:3]:  # Use first 3 available algorithms
            cache.set(test_file, f"{test_hash}_{algo.value}", algo)

        # Get all values to verify they're cached
        cached_values = {}
        for algo in list(HashAlgorithm.get_available())[:3]:
            cached_values[algo] = cache.get(test_file, algo)
            assert cached_values[algo] is not None

        # Invalidate one specific algorithm
        target_algo = list(HashAlgorithm.get_available())[0]
        cache.invalidate(test_file, target_algo)

        # Verify that only the target algorithm was invalidated
        assert cache.get(test_file, target_algo) is None
        for algo in list(HashAlgorithm.get_available())[1:3]:
            assert cache.get(test_file, algo) == cached_values[algo]

    def test_cache_invalidate_all_algorithms(self, test_file):
        """Test invalidating all algorithm cache entries for a file."""
        cache = HashCache()
        test_hash = "test_hash_value"

        # Add entries for multiple algorithms
        for algo in list(HashAlgorithm.get_available())[:3]:  # Use first 3 available algorithms
            cache.set(test_file, f"{test_hash}_{algo.value}", algo)

        # Get all values to verify they're cached
        for algo in list(HashAlgorithm.get_available())[:3]:
            assert cache.get(test_file, algo) is not None

        # Invalidate all algorithms
        cache.invalidate(test_file)

        # Verify that all algorithms were invalidated
        for algo in list(HashAlgorithm.get_available())[:3]:
            assert cache.get(test_file, algo) is None

    def test_cache_clear(self, test_file):
        """Test clearing all cache entries."""
        cache = HashCache()
        test_hash = "test_hash_value"

        # Add entries for multiple algorithms and files
        for algo in list(HashAlgorithm.get_available())[:2]:
            cache.set(test_file, f"{test_hash}_{algo.value}", algo)

            # Create another file
            other_file = test_file.parent / f"other_{algo.value}.txt"
            with open(other_file, "wb") as f:
                f.write(b"Other content")
            cache.set(other_file, f"other_{test_hash}_{algo.value}", algo)

        # Clear the cache
        cache.clear()

        # Verify all entries are gone
        for algo in list(HashAlgorithm.get_available())[:2]:
            assert cache.get(test_file, algo) is None

    def test_cache_size_limit(self, temp_dir):
        """Test that the cache respects its size limit."""
        max_size = 5
        cache = HashCache(max_size=max_size)

        # Create and cache more entries than the max size
        for i in range(max_size + 3):
            file_path = temp_dir / f"file{i}.txt"
            with open(file_path, "wb") as f:
                f.write(f"Content {i}".encode())

            # Add with different algorithms to exceed the limit faster
            for algo in list(HashAlgorithm.get_available())[:2]:
                cache.set(file_path, f"hash{i}_{algo.value}", algo)

                # If we've added enough entries, check that the cache size is respected
                if i >= 3:
                    assert len(cache._cache) <= max_size


class TestCachedChecksum:
    """Tests for the CachedChecksum wrapper class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield Path(tmpdirname)

    @pytest.fixture
    def test_file(self, temp_dir):
        """Create a test file."""
        file_path = temp_dir / "test.txt"
        with open(file_path, "wb") as f:
            f.write(b"Test content for cached checksum")
        return file_path

    def setup_method(self):
        """Setup method run before each test."""
        # Clear the cache before each test
        CachedChecksum.clear_cache()

    def test_cached_vs_uncached(self, test_file):
        """Test that cached and uncached checksums match with different algorithms."""
        for algo in list(HashAlgorithm.get_available())[:3]:
            # Calculate checksum without cache
            uncached = Checksum(test_file, hash_algorithm=algo)

            # Calculate checksum with cache
            cached = CachedChecksum(test_file, hash_algorithm=algo)

            # Checksums should match
            assert cached.checksum == uncached.checksum

    def test_cache_reuse(self, test_file):
        """Test that the cache is reused for different algorithm requests."""
        for algo in list(HashAlgorithm.get_available())[:3]:
            # First calculation (not cached)
            first = CachedChecksum(test_file, hash_algorithm=algo)

            # Second calculation (should use cache)
            second = CachedChecksum(test_file, hash_algorithm=algo)

            # Checksums should match
            assert first.checksum == second.checksum

            # Ideally, we'd verify that the second calculation actually used the cache
            # This is hard to test directly without mocking or instrumenting the code

    def test_file_modification_detection(self, temp_dir):
        """Test that modifications to a file invalidate the cache for all algorithms."""
        file_path = temp_dir / "changing.txt"

        # Create initial file
        with open(file_path, "wb") as f:
            f.write(b"Initial content")

        # Calculate initial checksums for multiple algorithms
        initial_checksums = {}
        for algo in list(HashAlgorithm.get_available())[:3]:
            initial = CachedChecksum(file_path, hash_algorithm=algo)
            initial_checksums[algo] = initial.checksum

        # Sleep to ensure modification time changes
        time.sleep(0.01)

        # Modify the file
        with open(file_path, "wb") as f:
            f.write(b"Modified content")

        # Calculate checksums after modification
        # Should detect the change and recompute
        for algo in list(HashAlgorithm.get_available())[:3]:
            modified = CachedChecksum(file_path, hash_algorithm=algo)
            # Checksums should be different
            assert initial_checksums[algo] != modified.checksum

    def test_cache_disabled(self, test_file):
        """Test using CachedChecksum with caching disabled for different algorithms."""
        for algo in list(HashAlgorithm.get_available())[:3]:
            # First calculation with cache
            cached = CachedChecksum(test_file, hash_algorithm=algo)

            # Second calculation with cache disabled
            uncached = CachedChecksum(test_file, hash_algorithm=algo, use_cache=False)

            # Checksums should match
            assert cached.checksum == uncached.checksum

    def test_cache_clearing(self, test_file):
        """Test that clearing the cache works for all algorithms."""
        # Calculate checksums for multiple algorithms (will be cached)
        original_checksums = {}
        for algo in list(HashAlgorithm.get_available())[:3]:
            checksum = CachedChecksum(test_file, hash_algorithm=algo)
            original_checksums[algo] = checksum.checksum

        # Modify the file without changing its mtime or size
        # This would normally not be detected by the cache key
        time.sleep(0.01)  # Sleep to ensure modification time changes
        original_content = test_file.read_bytes()
        with open(test_file, "wb") as f:
            # Write different content of same length
            modified_content = original_content.replace(b"Test", b"Best")
            f.write(modified_content)

        # Force the original mtime to simulate a change that wouldn't affect the cache key
        original_stat = test_file.stat()
        os.utime(test_file, (original_stat.st_atime, original_stat.st_mtime - 1.0))

        # Without clearing the cache, we'd get the cached checksums
        cached_results = {}
        for algo in list(HashAlgorithm.get_available())[:3]:
            cached = CachedChecksum(test_file, hash_algorithm=algo)
            cached_results[algo] = cached.checksum
            # With our manipulation, the cached result might match the original
            # due to the unchanged mtime and size

        # Clear the cache
        CachedChecksum.clear_cache()

        # Now we should get fresh checksums
        for algo in list(HashAlgorithm.get_available())[:3]:
            # Calculate without using cache
            uncached = Checksum(test_file, hash_algorithm=algo)
            # Calculate after clearing cache
            fresh = CachedChecksum(test_file, hash_algorithm=algo)
            # These should match each other
            assert fresh.checksum == uncached.checksum
            # And should be different from original if the content actually changed
            # which we can verify by checking uncached vs original
            if original_checksums[algo] != uncached.checksum:
                assert fresh.checksum != original_checksums[algo]

    def test_compute_hash_class_method(self, test_file):
        """Test the compute_hash class method with caching."""
        for algo in list(HashAlgorithm.get_available())[:3]:
            # First call computes and caches
            first_hash = CachedChecksum.compute_hash(test_file, algo)

            # Second call should use cache
            second_hash = CachedChecksum.compute_hash(test_file, algo)

            # Should be the same
            assert first_hash == second_hash

            # Call without cache
            uncached_hash = CachedChecksum.compute_hash(test_file, algo, use_cache=False)

            # Still should be the same
            assert first_hash == uncached_hash

    def test_legacy_sha256_method(self, test_file):
        """Test the legacy sha256 class method with caching."""
        # First call computes and caches
        first_hash = CachedChecksum.sha256(test_file)

        # Second call should use cache
        second_hash = CachedChecksum.sha256(test_file)

        # Should be the same
        assert first_hash == second_hash

        # Call without cache
        uncached_hash = CachedChecksum.sha256(test_file, use_cache=False)

        # Still should be the same
        assert first_hash == uncached_hash

    def test_hash_data_with_algorithms(self):
        """Test the hash_data method with different algorithms."""
        test_data = b"Test data for hashing with multiple algorithms"

        for algo in list(HashAlgorithm.get_available())[:3]:
            # Compute with enum
            hash1 = CachedChecksum.hash_data(test_data, algo)

            # Compute with string
            hash2 = CachedChecksum.hash_data(test_data, algo.value)

            # Compute with hashlib directly
            h = getattr(hashlib, algo.value)(test_data)
            expected_hash = h.hexdigest()

            # All should match
            assert hash1 == hash2 == expected_hash

    def test_invalidate_cache_entry_specific_algorithm(self, test_file):
        """Test invalidating cache for a specific file and algorithm."""
        # Compute checksums for multiple algorithms
        checksums = {}
        for algo in list(HashAlgorithm.get_available())[:3]:
            checksums[algo] = CachedChecksum.compute_hash(test_file, algo)

        # Invalidate a specific algorithm
        target_algo = list(HashAlgorithm.get_available())[0]
        CachedChecksum.invalidate_cache_entry(test_file, target_algo)

        # The invalidated algorithm should recompute
        new_hash = CachedChecksum.compute_hash(test_file, target_algo)
        assert new_hash == checksums[target_algo]  # Value should be the same

        # Other algorithms should still be cached
        for algo in list(HashAlgorithm.get_available())[1:3]:
            cached_hash = CachedChecksum.compute_hash(test_file, algo)
            assert cached_hash == checksums[algo]

    def test_invalidate_cache_entry_all_algorithms(self, test_file):
        """Test invalidating cache for all algorithms of a file."""
        # Compute checksums for multiple algorithms
        checksums = {}
        for algo in list(HashAlgorithm.get_available())[:3]:
            checksums[algo] = CachedChecksum.compute_hash(test_file, algo)

        # Invalidate all algorithms
        CachedChecksum.invalidate_cache_entry(test_file)

        # All algorithms should recompute but have the same values
        for algo in list(HashAlgorithm.get_available())[:3]:
            new_hash = CachedChecksum.compute_hash(test_file, algo)
            assert new_hash == checksums[algo]