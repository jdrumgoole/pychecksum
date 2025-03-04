from enum import Enum
from pathlib import Path
from typing import Optional
import json
from rich.table import Table
import csv
from rich import print

class OutputType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CSV = "csv"
    JSON = "json"


class OutputFormat:

    def __init__(self, f:OutputType):
        self._format = f

    @property
    def format(self):
        return self._format

    def output_row(self,filename:Path, algorithm:str, checksum:str, size:int):
        pass

    def print(self):
        pass

class OutputText(OutputFormat):

    def __init__(self):
        super().__init__(OutputType.TEXT)

    def output_row(self,filename:Path, algorithm:str, checksum:str, size:int):
        print(f"{algorithm}: {checksum} {filename.path} ({size})")


class OutputTable(OutputFormat):

    def __init__(self):
        super().__init__(OutputType.TABLE)
        # Create a rich table
        self._table = Table(title="File Checksums")
        self._table.add_column("File", style="cyan")
        self._table.add_column("Algorithm", style="green")
        self._table.add_column("Checksum", style="yellow")
        self._table.add_column("Size", style="blue")

    def table(self):
        return self._table

    def output_row(self,filename:Path, algorithm:str, checksum:str, size:int):
        self._table.add_row(str(filename), algorithm, checksum, size)

    def print(self):
        print(self._table)

class OutputFile(OutputFormat):

    def __init__(self, f:OutputType, output_filename: Optional[Path]):
        super().__init__(f)
        self._output_filename = output_filename
        self._output_file = None

    @property
    def output_filename(self):
        return self._output_filename

    def print(self):
        self._output_file.flush()
        with open(self._output_filename, "r") as f:
            for line in f:
                print(line, end="")

class OutputCSV(OutputFile):
        def __init__(self, output_file: Optional[Path]):
            super().__init__(OutputType.CSV, output_file)
            self._output_file = open(output_file, "w", newline="")
            self._writer = csv.writer(self._output_file)
            self._writer.writerow(["File", "Algorithm", "Checksum", "Size", "Size_Formatted"])

        def output_row(self, filename:Path, algorithm:str, checksum:str, size:int):
            self._writer.writerow([filename.path, algorithm, checksum, size])

        def close(self):
            self._output_file.close()

        def __delete__(self):
            self._output_file.close()

class OutputJSON(OutputFile):
    def __init__(self, output_file: Optional[Path]):
        super().__init__(OutputType.JSON, output_file)
        self._output_file = open(output_file, "w")

    def output_row(self, filename:Path, algorithm:str, checksum:str, size:int):
        json.dump({"File":filename.path, "Algorithm":algorithm, "Checksum":checksum, "Size":size}, self._output_file, indent=2)

class OutputFactory:

    @staticmethod
    def make_output_object(fmt: OutputType, output_file: Optional[Path]) -> OutputFile:
        match fmt:
            case OutputType.TABLE:
                return OutputTable()
            case OutputType.CSV:
                return OutputCSV(output_file)
            case OutputType.JSON:
                return OutputJSON(output_file)
            case OutputType.TEXT:
                return OutputText()

    @staticmethod
    def print(o:OutputFile):
        o.print()




