from dataclasses import dataclass
from pathlib import Path

@dataclass
class WorkerConfig:
    address: str
    database_type: str
    runtime: str
    max_workers: int
    max_rpcs: int
    timeout: int
    auto_remove: bool
    log_level: str
    log_format: str
    log_file: str | None
    containerized: bool
    update_buffer_size: int
    db_address: str
    models: dict[str, Path]
