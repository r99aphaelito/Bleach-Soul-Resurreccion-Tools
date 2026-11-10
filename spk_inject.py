#!/usr/bin/env python3
"""
CREDITS: XANDREALE FROM OLD XENTAX FORUM

spk_inject.py - Reinjects edited .dds textures into an Img.spk/Tex.spk

Workflow
-----------------
1. Extract the textures:
       python spk_extractor.py figureXxxImg.spk
      -> generates figureXxxImg_1.dds, figureXxxImg_2.dds, ...

2. Edit whichever .dds files you want (see the rules below).

3. Reinject:
       python spk_inject.py figureXxxImg.spk
      -> generates figureXxxImg_patched.spk (the original is NEVER overwritten)

4. Replace the game's original .spk with the _patched file (using the original filename).

Rules for edited .dds files (otherwise the script will reject the texture)
------------------------------------------------------------
- Same width and height.
- Same compression format (DXT1, DXT3, or DXT5). Format 133 textures are
  uncompressed and are extracted/edited as PNG (name_N.png, needs Pillow).
- Must include mipmaps (the same number as the original), so the data size
  is IDENTICAL to the original.
- Legacy format (without a "DX10" header).

The file is modified "in place": since the .spk size does not change, no
other game offsets/tables need to be updated. Format
166 (linear) textures are automatically converted back to the original layout
automatically.

Usage
---
    python spk_inject.py original.spk [dds_folder] [output.spk]
"""

import os
import sys

from spk_extractor import (
    FMT_ARGB8888_SWIZZLED,
    TEXTURE_FORMATS,
    bgra_to_argb_swizzled,
    parse_dds_argb,
    find_texture_archives,
    parse_dds,
    repack_texture,
    unpack_texture,
)


def inject(spk_path: str, dds_dir: str, out_path: str) -> None:
    data = bytearray(open(spk_path, "rb").read())
    stem = os.path.splitext(os.path.basename(spk_path))[0]

    texture_number = 0  # same numbering used by the extractor
    replaced = 0

    for start, infos in find_texture_archives(bytes(data)):
        for info in infos:
            texture_number += 1
            dds_path = os.path.join(dds_dir, f"{stem}_{texture_number}.dds")
            png_path = os.path.join(dds_dir, f"{stem}_{texture_number}.png")

            if info.fmt == FMT_ARGB8888_SWIZZLED:
                # Uncompressed texture: edited as PNG (DDS A8R8G8B8 also accepted)
                src = png_path if os.path.exists(png_path) else dds_path
                if not os.path.exists(src):
                    continue
                label = f"#{texture_number} ({os.path.basename(src)})"
                try:
                    if src == png_path:
                        from PIL import Image
                        im = Image.open(src).convert("RGBA")
                        width, height = im.size
                        bgra = im.tobytes("raw", "BGRA")
                    else:
                        with open(src, "rb") as f:
                            width, height, bgra = parse_dds_argb(f.read())
                except Exception as err:
                    print(f"[SKIPPED] {label}: {err}")
                    continue
                if (width, height) != (info.width, info.height):
                    print(f"[SKIPPED] {label}: size {width}x{height}, expected {info.width}x{info.height}")
                    continue
                region = slice(start + info.offset, start + info.offset + info.size)
                new_raw = bgra_to_argb_swizzled(bgra, width, height)
                if len(new_raw) != info.size:
                    print(f"[SKIPPED] {label}: {len(new_raw)} bytes, expected {info.size}")
                elif new_raw == bytes(data[region]):
                    print(f"[UNCHANGED] {label}: no changes")
                else:
                    data[region] = new_raw
                    replaced += 1
                    print(f"[OK     ] {label}: replaced (ARGB8888)")
                continue

            if not os.path.exists(dds_path):
                continue

            label = f"#{texture_number} ({os.path.basename(dds_path)})"

            if info.fmt not in TEXTURE_FORMATS:
                print(f"[SKIPPED] {label}: unknown format {info.fmt}")
                continue

            try:
                with open(dds_path, "rb") as f:
                    width, height, fourcc, pixels = parse_dds(f.read())
            except ValueError as err:
                print(f"[SKIPPED] {label}: {err}")
                continue

            region = slice(start + info.offset, start + info.offset + info.size)
            original = bytes(data[region])
            original_packed, _levels = unpack_texture(info, original)
            expected_fourcc = TEXTURE_FORMATS[info.fmt]

            if (width, height) != (info.width, info.height):
                print(f"[SKIPPED] {label}: size {width}x{height}, expected {info.width}x{info.height}")
            elif fourcc != expected_fourcc:
                print(f"[SKIPPED] {label}: formato {fourcc}, expected {expected_fourcc}")
            elif len(pixels) != len(original_packed):
                print(
                    f"[SKIPPED] {label}: {len(pixels)} bytes de dados, expected {len(original_packed)}. "
                    "Generate mipmaps using the same count as the original."
                )
            else:
                new_raw = repack_texture(info, pixels, original)
                if new_raw == original:
                    print(f"[UNCHANGED] {label}: no changes")
                else:
                    data[region] = new_raw
                    replaced += 1
                    print(f"[OK     ] {label}: replaced")

    with open(out_path, "wb") as f:
        f.write(data)
    print(f"\n{replaced} textura(s) replaced(s). Gravado: {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python spk_inject.py original.spk [dds_folder] [output.spk]")
        sys.exit(1)

    spk_path = sys.argv[1]
    dds_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    stem = os.path.splitext(os.path.basename(spk_path))[0]
    out_path = sys.argv[3] if len(sys.argv) > 3 else f"{stem}_patched.spk"

    if os.path.abspath(out_path) == os.path.abspath(spk_path):
        print("The output cannot be the same file as the original.")
        sys.exit(1)

    inject(spk_path, dds_dir, out_path)


if __name__ == "__main__":
    main()
