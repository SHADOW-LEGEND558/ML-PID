"""Verify historical text-file hashes across Git line-ending conversions."""
import hashlib
from pathlib import Path


def matches_recorded_hash(path, expected):
    """Accept the recorded bytes or their LF/CRLF equivalent, never data edits.

    Original study hashes describe Windows working files. Git stores LF text,
    so a clean checkout can have different bytes with identical CSV values.
    Binary files always require an exact byte match.
    """
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return True
    if path.suffix.lower() not in {'.csv', '.json', '.py', '.md', '.txt', '.log'}:
        return False
    lf = raw.replace(b'\r\n', b'\n')
    return any(hashlib.sha256(data).hexdigest() == expected
               for data in (lf, lf.replace(b'\n', b'\r\n')))
