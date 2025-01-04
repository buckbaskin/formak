from dataclasses import dataclass
from typing import Optional


@dataclass
class CppCompileResult:
    success: bool
    header_path: Optional[str] = None
    source_path: Optional[str] = None
