from __future__ import annotations

from email.message import Message


def parse_header(line: str) -> tuple[str, dict[str, str]]:
    message = Message()
    message["content-type"] = line
    return message.get_content_type(), dict(message.get_params()[1:])
