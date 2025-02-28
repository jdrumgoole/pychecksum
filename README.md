# Checksum

A flexible, high-performance Python package for calculating and caching file checksums using any hash algorithm supported by hashlib.

## Note for source code readers
We use [invoke](https://docs.pyinvoke.org/en/stable/) to drive a bunch of internal tasks. 
You can see the list of tasks in `tasks.py` at the top level. 

## Features

- **Multiple Algorithm Support**: Use any hash algorithm available in hashlib (MD5, SHA1, SHA256, SHA512, etc.)
- **Type-Safe Enum Interface**: HashAlgorithm enum for better type safety and IDE auto-completion
- **Performance Optimized**: Configurable block sizes for reading large files efficiently
- **Smart Caching**: Built-in caching system based on file modification times and content
- **Low Memory Footprint**: Stream-based processing keeps memory usage low, even for very large files
- **Comprehensive CLI**: Powerful command-line interface with recursive directory processing and multiple output formats
- **full cli support**: The package includes a powerful command-line tool (clichecksum) for computing and verifying checksums.
## Installation

```bash
# Using pip
pip install pychecksum

# From source
git clone https://github.com/yourusername/checksum.git
cd checksum
pip install -e .
```

## Basic Usage

### Computing Checksums

```python
from src.pychecksum import Checksum, HashAlgorithm

# Calculate a SHA-256 checksum (default)
checksum = Checksum("myfile.txt")
print(f"SHA-256: {checksum.checksum}")

# Calculate with a different algorithm
md5_checksum = Checksum("myfile.txt", hash_algorithm=HashAlgorithm.MD5)
print(f"MD5: {md5_checksum.checksum}")

# Calculate with a custom block size (for large files)
large_file_checksum = Checksum("largefile.iso", block_size=1048576, hash_algorithm=HashAlgorithm.SHA512)
print(f"SHA-512: {large_file_checksum.checksum}")
```

### Using the Cache

```python
from src.pychecksum import CachedChecksum, HashAlgorithm

# First calculation computes and caches
cached = CachedChecksum("myfile.txt", hash_algorithm=HashAlgorithm.SHA256)
print(f"SHA-256: {cached.checksum}")

# Second calculation uses the cache if the file hasn't changed
cached2 = CachedChecksum("myfile.txt", hash_algorithm=HashAlgorithm.SHA256)
print(f"SHA-256 (from cache): {cached2.checksum}")

# Force a fresh calculation
no_cache = CachedChecksum("myfile.txt", hash_algorithm=HashAlgorithm.SHA256, use_cache=False)
print(f"SHA-256 (fresh): {no_cache.checksum}")

# Using a different algorithm creates a separate cache entry
md5_cached = CachedChecksum("myfile.txt", hash_algorithm=HashAlgorithm.MD5)
print(f"MD5: {md5_cached.checksum}")
```

### Using Static Methods

```python
from src.pychecksum import CachedChecksum, HashAlgorithm

# Static method for computing hash with caching
sha256_hash = CachedChecksum.compute_hash("myfile.txt", HashAlgorithm.SHA256)
print(f"SHA-256: {sha256_hash}")

# Hash in-memory data
data = b"Hello, World!"
data_hash = CachedChecksum.hash_data(data, HashAlgorithm.SHA256)
print(f"SHA-256 of data: {data_hash}")
```

### Available Hash Algorithms

```python
from src.pychecksum import HashAlgorithm

# List all available algorithms
available_algos = HashAlgorithm.get_available()
print("Available algorithms:", [algo.value for algo in available_algos])

# Check if an algorithm is available
if HashAlgorithm.is_available(HashAlgorithm.BLAKE2B):
    print("BLAKE2B is available")
else:
    print("BLAKE2B is not available")
```

## Command Line Interface

The package includes a powerful command-line tool for computing and verifying checksums.

### Generating Checksums

```bash
# Basic usage with default SHA-256
checksum hash myfile.txt

# Specify a different algorithm
checksum hash myfile.txt --algorithm md5

# Generate multiple hashes at once
checksum hash myfile.txt --multi md5 --multi sha1 --multi sha256

# Process multiple files
checksum hash file1.txt file2.txt file3.txt

# Process directories recursively
checksum hash /path/to/directory --recursive

# Exclude files/directories
checksum hash /path/to/directory --recursive --exclude "*.tmp" --exclude "*cache*"

# Change output format
checksum hash myfile.txt --format json

# Save results to a file
checksum hash myfile.txt --output results.json --format json
```

### Verifying Checksums

```bash
# Verify a file against a checksum
checksum verify myfile.txt abc123def456...

# Specify algorithm
checksum verify myfile.txt abc123def456... --algorithm md5

# Batch verify from a checksums file
checksum batch checksums.txt

# Specify a base directory for relative paths in batch file
checksum batch checksums.txt --base-dir /path/to/files
```

### Getting Help

```bash
# List all commands
checksum --help

# Command-specific help
checksum hash --help
checksum verify --help
checksum batch --help

# List available hash algorithms
checksum hash --list-algorithms
```

## API Reference

### Core Classes

- **HashAlgorithm**: Enum of supported hash algorithms
- **Checksum**: Base class for computing file checksums
- **HashCache**: Class for caching hash values
- **CachedChecksum**: Wrapper that adds caching to Checksum operations

### HashAlgorithm Enum

| Member   | Value     |
|----------|-----------|
| MD5      | 'md5'     |
| SHA1     | 'sha1'    |
| SHA224   | 'sha224'  |
| SHA256   | 'sha256'  |
| SHA384   | 'sha384'  |
| SHA512   | 'sha512'  |
| BLAKE2B  | 'blake2b' |
| BLAKE2S  | 'blake2s' |
| SHA3_224 | 'sha3_224'|
| SHA3_256 | 'sha3_256'|
| SHA3_384 | 'sha3_384'|
| SHA3_512 | 'sha3_512'|

## Performance Tips

- For large files, a larger block size (e.g., 1MB = 1048576 bytes) can improve performance
- For many small files, using the default block size is generally optimal
- The cache dramatically improves performance when checking the same files multiple times
- When verifying a large number of files, use the `batch` command with `--parallel` for multi-threading

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.