"""
Checksum CLI - A command line tool for generating file checksums with multiple algorithms.

This tool uses the checksum package to generate checksums for files using any
algorithm supported by hashlib.
"""

import sys
import typer
from enum import Enum
from typing import List, Optional, Generator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich import print

from pychecksumtool import HashAlgorithm, CachedChecksum
from pychecksumtool.scripts.clioutput import OutputFactory, OutputType

# Create Typer app
app = typer.Typer(
    help="A command line tool for generating file checksums with multiple algorithms.",
    add_completion=True,
)

# Get the console for rich output
console = Console()

@app.command()
def hash(
    files: List[Path] = typer.Argument(
        ...,
        exists=True,
        help="One or more files or directories to compute checksums for",
        show_default=False,
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Process directories recursively",
    ),
    exclude: List[str] = typer.Option(
        [],
        "--exclude",
        "-e",
        help="Exclude files/directories matching these patterns (can be specified multiple times)",
    ),
    algorithm: str = typer.Option(
        "sha256",
        "--algorithm",
        "-a",
        help="Hash algorithm to use (e.g., md5, sha1, sha256)",
    ),
    block_size: int = typer.Option(
        65536,
        "--block-size",
        "-b",
        help="Block size in bytes for reading files",
    ),
    list_algorithms: bool = typer.Option(
        False,
        "--list-algorithms",
        "-l",
        help="List available hash algorithms and exit",
    ),
    format: OutputType= typer.Option(
        OutputType.TABLE,
        "--fmt",
        "-f",
        help="Output fmt",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write output to file instead of stdout",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable caching of checksums",
    ),
    delay: float = typer.Option(
        None,
        help="delay between each block read in seconds, stops process saturating the I/O or CPU",
    ),
    multiple_algorithms: List[str] = typer.Option(
        [],
        "--multi",
        "-m",
        help="Compute multiple hash algorithms at once (can be specified multiple times)",
    ),
) -> None:
    """
    Compute checksums for one or more files using the specified algorithm.
    """
    # List available algorithms if requested
    if list_algorithms:
        show_available_algorithms()
        return

    # Process the algorithms
    algorithms = []

    # If multiple algorithms are specified, use those
    if multiple_algorithms:
        for algo in multiple_algorithms:
            try:
                algorithms.append(HashAlgorithm.from_string(algo))
            except ValueError as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                return
    # Otherwise use the single algorithm
    else:
        try:
            algorithms.append(HashAlgorithm.from_string(algorithm))
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return

    # Check that all algorithms are available


    for algo in algorithms:
        if not HashAlgorithm.is_available(algo):
            console.print(f"[bold red]Error:[/bold red] Algorithm {algo.value} is not available.")
            show_available_algorithms()
            return

    output_mode = OutputFactory.make_output_object(format, output_file)
    for file_path in track(collect_paths(files, recursive, exclude, verbose), description=f"Processing paths", update_period=0.1, refresh_per_second=5):

    # Process each file
    #for file_path in all_files:
        if verbose:
            print(f"Processing {file_path}", end=None)

        try:
            file_size = file_path.stat().st_size
            file_size_formatted = format_size(file_size)

            # Process each algorithm for this file
            for algo in algorithms:
                algorithm_name = algo.value

                if verbose:
                    print(f"[cyan]Computing [/cyan][yellow]{algorithm_name}[/yellow] [cyan]for[/cyan] [yellow]{file_path}[/yellow]")

                try:
                    # Compute checksum
                    checksum = CachedChecksum.compute_hash(
                        file_path,
                        hash_algorithm=algo,
                        block_size=block_size,
                        use_cache=not no_cache,
                        delay=delay
                    )
                    if verbose:
                        print(f"[cyan]{algorithm_name}:[/cyan] [yellow]{checksum}[/yellow]")


                except Exception as e:
                    print(f"[bold red]Error computing {algorithm_name} for {file_path}:[/bold red] {str(e)}")

                output_mode.output_row(file_path, algorithm_name, checksum, file_size_formatted)
        except Exception as e:
            print(f"[bold red]Error processing {file_path}:[/bold red] {str(e)}")

    output_mode.print()

@app.command()
def verify(
    file: Path = typer.Argument(..., exists=True, help="File to verify"),
    checksum: str = typer.Argument(..., help="Expected checksum value"),
    algorithm: str = typer.Option(
        "sha256",
        "--algorithm",
        "-a",
        help="Hash algorithm to use (e.g., md5, sha1, sha256)",
    ),
    block_size: int = typer.Option(
        65536,
        "--block-size",
        "-b",
        help="Block size in bytes for reading files",
    ),
    delay: float = typer.Option(
        None,
        help="delay between each block read in seconds, stops process saturating the I/O or CPU",
    ),
    ignore_case: bool = typer.Option(
        False,
        "--ignore-case",
        "-i",
        help="Ignore case when comparing checksums",
    ),
) -> None:
    """
    Verify that a file matches an expected checksum.
    """
    try:
        algo = HashAlgorithm.from_string(algorithm)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    if not HashAlgorithm.is_available(algo):
        console.print(f"[bold red]Error:[/bold red] Algorithm {algo.value} is not available.")
        show_available_algorithms()
        return

    try:
        actual_checksum = CachedChecksum.compute_hash(file, hash_algorithm=algo, block_size=block_size, delay=delay)

        # Compare checksums
        if ignore_case:
            expected = checksum.lower()
            actual = actual_checksum.lower()
        else:
            expected = checksum
            actual = actual_checksum

        if expected == actual:
            console.print(f"[bold green]✓ Checksum verified:[/bold green] {file}")
            return 0
        else:
            console.print(f"[bold red]✗ Checksum mismatch for {file}[/bold red]")
            console.print(f"  Expected: {checksum}")
            console.print(f"  Actual:   {actual_checksum}")
            return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 2


@app.command()
def batch(
    checksums_file: Path = typer.Argument(
        ...,
        exists=True,
        help="File containing checksums to verify (in fmt: CHECKSUM FILENAME)",
    ),
    algorithm: str = typer.Option(
        "sha256",
        "--algorithm",
        "-a",
        help="Hash algorithm to use (e.g., md5, sha1, sha256)",
    ),
    block_size: int = typer.Option(
        65536,
        "--block-size",
        "-b",
        help="Block size in bytes for reading files",
    ),
    ignore_case: bool = typer.Option(
        False,
        "--ignore-case",
        "-i",
        help="Ignore case when comparing checksums",
    ),
    base_dir: Optional[Path] = typer.Option(
        None,
        "--base-dir",
        "-d",
        help="Base directory for relative file paths",
    ),
    delay: float = typer.Option(
        None,
        help="delay between each block read in seconds, stops process saturating the I/O or CPU",
    ),
    parallel: bool = typer.Option(
        True,
        "--parallel/--sequential",
        help="Process files in parallel or sequentially",
    ),
) -> None:
    """
    Verify multiple file checksums from a batch file.

    The checksums file should be in the fmt:
    CHECKSUM FILENAME

    One entry per line, with the checksum and filename separated by whitespace.
    """
    try:
        algo = HashAlgorithm.from_string(algorithm)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    if not HashAlgorithm.is_available(algo):
        console.print(f"[bold red]Error:[/bold red] Algorithm {algo.value} is not available.")
        show_available_algorithms()
        return

    # Read checksums file
    entries = []
    try:
        with open(checksums_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Split line into checksum and filename
                parts = line.split(None, 1)
                if len(parts) != 2:
                    console.print(f"[yellow]Warning: Skipping invalid line: {line}[/yellow]")
                    continue

                checksum, filename = parts

                # Handle base directory
                if base_dir:
                    file_path = base_dir / filename
                else:
                    file_path = Path(filename)

                entries.append((checksum, file_path))
    except Exception as e:
        console.print(f"[bold red]Error reading checksums file:[/bold red] {str(e)}")
        return

    console.print(f"Verifying {len(entries)} files...")

    # Track results
    passed = 0
    failed = 0
    errors = 0

    # Process entries

        # Function to verify a single entry
    def verify_entry(entry):
        expected_checksum, file_path = entry

        if not file_path.exists():
            print(f"[yellow]File not found: {file_path}[/yellow]")
            return "error", f"File not found: {file_path}"

        try:
            actual_checksum = CachedChecksum.compute_hash(
                file_path,
                hash_algorithm=algo,
                block_size=block_size,
                delay=delay
            )

            # Compare checksums
            if ignore_case:
                expected = expected_checksum.lower()
                actual = actual_checksum.lower()
            else:
                expected = expected_checksum
                actual = actual_checksum

            if expected == actual:
                return "pass", None
            else:
                return "fail", (expected_checksum, actual_checksum)
        except Exception as e:
            return "error", str(e)

    # Process entries either in parallel or sequentially
    if parallel and len(entries) > 1:
        with ThreadPoolExecutor() as executor:
            for i, (entry, result) in enumerate(zip(entries, executor.map(verify_entry, entries))):
                _, file_path = entry
                status, details = result

                if status == "pass":
                    print(f"[green]✓ {file_path}[/green]")
                    passed += 1
                elif status == "fail":
                    expected, actual = details
                    print(f"[red]✗ {file_path}[/red]")
                    print(f"  Expected: {expected}")
                    print(f"  Actual:   {actual}")
                    failed += 1
                else:  # error
                    print(f"[yellow]! {file_path}: {details}[/yellow]")
                    errors += 1

    else:
        for entry in entries:
            expected_checksum, file_path = entry

            status, details = verify_entry(entry)

            if status == "pass":
                print(f"[green]✓ {file_path}[/green]")
                passed += 1
            elif status == "fail":
                expected, actual = details
                print(f"[red]✗ {file_path}[/red]")
                print(f"  Expected: {expected}")
                print(f"  Actual:   {actual}")
                failed += 1
            else:  # error
                print(f"[yellow]! {file_path}: {details}[/yellow]")
                errors += 1


    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  [green]Passed: {passed}[/green]")
    if failed > 0:
        console.print(f"  [red]Failed: {failed}[/red]")
    if errors > 0:
        console.print(f"  [yellow]Errors: {errors}[/yellow]")

    # Return appropriate exit code
    if failed > 0 or errors > 0:
        return 1
    else:
        return 0


def show_available_algorithms() -> None:
    """Show a list of available hash algorithms."""
    available = HashAlgorithm.get_available()

    table = Table(title="Available Hash Algorithms")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Available", style="green")

    # Add all algorithms, marking those that are available
    for algo in sorted(HashAlgorithm, key=lambda x: x.value):
        if algo in available:
            table.add_row(algo.value, "✓")
        else:
            table.add_row(algo.value, "✗", style="dim")

    console.print(table)


def format_size(size_bytes: int) -> str:
    """Format a size in bytes to a human-readable fmt."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.1f} GB"



def collect_paths(files: List[Path], recursive: bool, exclude_patterns: List[str], verbose=bool) -> Generator[Path, None, None]:
    """Collect all files to process."""
    for path in files:
        if path.is_file():
            yield path
        elif path.is_dir() and recursive:
            for file_path in collect_dirs(path, exclude_patterns, verbose):
                yield file_path
        elif path.is_dir():
            # If not recursive, just include files in the top directory
            for item in path.iterdir():
                if item.is_file() and not is_excluded(item, exclude_patterns):
                    yield item

def collect_dirs(directory: Path, exclude_patterns: List[str], verbose:bool) -> Generator[Path, None, None]:
    """Recursively collect all files in a directory, excluding any that match the patterns."""
    files = []

    try:
        # Handle case where directory doesn't exist or isn't accessible
        if not directory.exists() or not directory.is_dir():
            print(f"[yellow]Warning: {directory} is not a valid directory[/yellow]")
            return files

        for item in directory.iterdir():
            # Check if item should be excluded
            if is_excluded(item, exclude_patterns):
                if verbose:
                    print(f"Skipping excluded path: {item}")
                continue

            if item.is_file():
                yield item
                if verbose:
                    print(f"processing file: {item}")
            elif item.is_dir():
                # Recursively collect files from subdirectory
                yield from collect_dirs(item, exclude_patterns, verbose)
    except PermissionError:
        print(f"[yellow]Permission denied: Cannot access {directory}[/yellow]")
    except Exception as e:
        print(f"[yellow]Error accessing {directory}: {str(e)}[/yellow]")

    return files

def is_excluded(path: Path, exclude_patterns: List[str]) -> bool:
    """Check if a path should be excluded based on patterns."""
    if not exclude_patterns:
        return False

    path_str = str(path)
    for pattern in exclude_patterns:
        import fnmatch
        if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path_str, pattern):
            return True
    return False

def main():
    # need a main line function to define in pyproject.toml
    app()

if __name__ == "__main__":
    main()