#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory import MemoryCompressor, Scratchpad


def main():
    scratchpad = Scratchpad()
    compressor_struct = MemoryCompressor(strategy="structured", max_items=8)
    compressor_text = MemoryCompressor(strategy="free_text", max_items=8)

    sample_observations = [
        "You are in the kitchen. Exits are north and east.",
        "The door is locked. You need a brass key.",
        "You can't open the chest. Nothing happens.",
        "You take the brass key.",
        "You unlock the door and open it.",
    ]

    for obs in sample_observations:
        scratchpad.update(obs)

    structured = scratchpad.compress(compressor_struct)
    free_text = scratchpad.compress(compressor_text)

    payload = {
        "scratchpad": scratchpad.export(),
        "structured_compression": structured,
        "free_text_compression": free_text,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
