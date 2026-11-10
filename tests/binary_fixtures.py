"""Synthetic ELF/OM structures for deployment boundary tests."""

import struct


def device_elf():
    """A minimal nonempty NPU ELF for the binary portability parser only."""
    raw = bytearray(208)
    raw[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHIQQQIHHHHHH", raw, 16, 1, 4137, 1, 0, 0, 80, 0, 64, 0, 0, 64, 2, 0)
    raw[64:80] = bytes(range(1, 17))
    struct.pack_into("<IIQQQQIIQQ", raw, 144, 0, 1, 6, 0, 64, 16, 0, 0, 16, 0)
    return bytes(raw)


def standard_om(parts):
    """Use GE's public standard OM header and aligned wide partition table."""
    header = bytearray(256)
    struct.pack_into("<III", header, 0, 0x444F4D49, 256, 0x20000000)
    header[82] = 1
    struct.pack_into("<I", header, 156, 1)
    table = bytearray(8 + 24 * len(parts))
    struct.pack_into("<I", table, 0, len(parts))
    offset = 0
    for index, (kind, payload) in enumerate(parts):
        struct.pack_into("<I4xQQ", table, 8 + 24 * index, kind, offset, len(payload))
        offset += len(payload)
    struct.pack_into("<Q", header, 184, len(table) + offset)
    return bytes(header + table) + b"".join(payload for _, payload in parts)
