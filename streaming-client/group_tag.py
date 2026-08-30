"""The metadata group tag: this client's format for fasth3 clip metadata.

The director writes it at enqueue time; the overlay and the director's own
narration read it back off the metadata echo. It lives in its own module
because both ends of the pipeline need it and neither should import the
other (the director sits upstream of the link, the overlay downstream of
the pacer — a shared import in either direction is a cycle).

The format itself is JSON with a `group_id` plus title, scene numbering,
author, source, `generated`, and the truncated raw prompt — see the
director's `_enqueue_group` for the authoritative writer.
"""

from __future__ import annotations

import json


def parse_group_tag(metadata: str) -> dict | None:
    """Read this client's group tag back out of a clip's metadata echo.

    Returns None for metadata this client did not write (other clients'
    clips, or an empty string).
    """
    try:
        tag = json.loads(metadata)
    except (TypeError, ValueError):
        return None
    if not isinstance(tag, dict) or "group_id" not in tag:
        return None
    return tag
