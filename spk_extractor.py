#!/usr/bin/env python3
"""
CREDITS: XANDREALE FROM OLD XENTAX FORUM

spk_extractor.py - SPK file extractor (meshes and textures)

Python translation of the FBSL script "Xandreale modeler exporter", with
fixes made after testing with real files (btlIchigoDokuro*.spk).

What the script does
------------------
1. MESH files (any .spk that does NOT end with "Img.spk"/"Tex.spk")
      -> scanned for mesh blocks and exported as .obj.

2. TEXTURE files (ending in "Img.spk" or "Tex.spk")
      -> textures (DXT1/DXT3/DXT5) are extracted and saved as .dds.

All file data is BIG-ENDIAN.

Usage
---
    python spk_extractor.py arquivo.spk

Output files are written to the current directory, using the .spk filename
(without its extension) as the prefix.

Fixes compared to the first version
--------------------------------------
- Textures: the first dimension field is the WIDTH (it was previously reversed,
  which corrupted non-square textures).
- Texturas formato 166 (DXT1 "linear"): os mipmaps ficam empilhados numa
  area with a fixed pitch; they are now converted to standard DDS layout.
- Format 135 is DXT3 (visually confirmed by the alpha channel).
- Meshes: the original `checkit` filter discarded valid triangles
  (any triangle lying on an axis-aligned plane). Now only
  truly degenerate triangles (zero area) are discarded.
- Block searching uses `find` + validation instead of relying on
  sequential read alignment.
"""

import os
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Signatures (read as big-endian) marking the start of each mesh block.
# In the original FBSL, these appear as 16777602 and 33554818 (read as little-endian).
MESH_TAG = 0x82010001        # simple mesh (1 group)
SPLIT_MESH_TAG = 0x82010002  # split mesh (2 groups: A and B)

# Signature of an embedded texture archive (33620479 in FBSL)
TEXTURE_ARCHIVE_TAG = 0x020101FF

# Texture format (table's "fmt" byte) -> DDS FourCC.
# The values appear to be PS3 GCM formats: 0x86 = DXT1, 0x87 = DXT2/3,
# 0x88 = DXT4/5, and bit 0x20 (0xA6) indicates a "linear" texture (with pitch).
TEXTURE_FORMATS = {
    134: "DXT1",
    166: "DXT1",  # DXT1 linear: mipmaps empilhados com pitch fixo
    135: "DXT3",
    136: "DXT5",
}

# DDS header values used by the original script
DDS_FLAGS = 659463        # 0x000A1007 = CAPS|HEIGHT|WIDTH|PIXELFORMAT|MIPMAPCOUNT|LINEARSIZE
DDS_PF_FOURCC = 4         # DDPF_FOURCC: the format is specified by a FourCC
DDS_CAPS = 4198408        # 0x00401008 = COMPLEX|TEXTURE|MIPMAP

# Limits used to validate candidates to file-intern textures
MAX_TEXTURES_PER_ARCHIVE = 1024
MAX_TEXTURE_DIMENSION = 8192


# ----------------------------------------------------------------------------
# Helper functions for reading (big-endian)
# ----------------------------------------------------------------------------

def read_u32(data: bytes, pos: int) -> int:
    """Reads an unsigned 32-bit integer (big-endian). Returns 0 if out of bounds."""
    if pos < 0 or pos + 4 > len(data):
        return 0
    return struct.unpack_from(">I", data, pos)[0]


def read_u16_array(data: bytes, pos: int, count: int) -> List[int]:
    """Reads `count` 16-bit integers (big-endian) starting at `pos`."""
    return list(struct.unpack_from(f">{count}H", data, pos))


def read_float_array(data: bytes, pos: int, count: int) -> List[float]:
    """Reads `count` 32-bit floats (big-endian) starting at `pos`."""
    return list(struct.unpack_from(f">{count}f", data, pos))


def read_half_array(data: bytes, pos: int, count: int) -> List[float]:
    """Reads `count` 16-bit half-floats (big-endian) and converts them to floats."""
    return list(struct.unpack_from(f">{count}e", data, pos))


def find_all(data: bytes, pattern: bytes):
    """Yields all positions (at any alignment) of `pattern` in `data`."""
    pos = data.find(pattern)
    while pos != -1:
        yield pos
        pos = data.find(pattern, pos + 1)


# ----------------------------------------------------------------------------
# MESHES
# ----------------------------------------------------------------------------

Vertex = Tuple[float, float, float]
UV = Tuple[float, float]


def choose_uv_set(candidates: List[List[UV]], indices: List[int]) -> List[UV]:
    """
    Selects which UV set to use when each vertex has more than one pair.

    Meshes with 8 bytes of UV data per vertex store two pairs of half-floats.
    The header does not indicate which pair contains the texture UVs (in some meshes,
    the first pair contains other data that becomes NaN/huge values).
    Heuristic: discard sets containing NaN/out-of-range values and choose the one with
    the lowest average UV distance between vertices connected by an edge (real UVs
    vary smoothly across the mesh).
    With a single candidate (stride 4), it is always selected.
    """
    if len(candidates) == 1:
        return candidates[0]

    def cost(uvs: List[UV]) -> float:
        if any(x != x or abs(x) > 100 for uv in uvs for x in uv):
            return float("inf")  # NaN or out-of-range value
        total, count = 0.0, 0
        for i in range(0, len(indices) - 2, 3):
            tri = indices[i:i + 3]
            if max(tri) >= len(uvs):
                continue
            for p, q in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                total += abs(uvs[p][0] - uvs[q][0]) + abs(uvs[p][1] - uvs[q][1])
                count += 1
        return total / count if count else 0.0

    return min(candidates, key=cost)


def read_mesh_data(
    data: bytes,
    index_offset: int,
    vertex_offset: int,
    uv_offset: int,
    has_uvs: bool,
    face_index_count: int,
    vertex_count: int,
    uv_stride: int = 4,
):
    """
    Reads the three mesh arrays: indices, vertices, and UVs.

    Data layout (all big-endian):
      - indices: `face_index_count` words (uint16), 3 indices per triangle
      - vertices: `vertex_count` * (float x, float y, float z)
      - UVs: `vertex_count` * `uv_stride` bytes, stored as half-floats.
                  stride 4 = (u, v); stride 8 = two pairs of half-floats per
                  vertex (see `choose_uv_set`).

    Returns (indices, vertices, uvs, final_read_position).
    """
    indices = read_u16_array(data, index_offset, face_index_count)

    flat = read_float_array(data, vertex_offset, vertex_count * 3)
    vertices: List[Vertex] = [
        (flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)
    ]
    end_pos = vertex_offset + vertex_count * 12

    # UVs are optional. Without UVs, (0, 0) is used for all vertices.
    if has_uvs:
        halves_per_vertex = max(2, uv_stride // 2)
        flat_uv = read_half_array(data, uv_offset, vertex_count * halves_per_vertex)
        # One candidate per pair of half-floats within each vertex
        candidates = [
            [(flat_uv[v * halves_per_vertex + k], flat_uv[v * halves_per_vertex + k + 1])
             for v in range(vertex_count)]
            for k in range(0, halves_per_vertex, 2)
        ]
        uvs: List[UV] = choose_uv_set(candidates, indices)
        end_pos = uv_offset + vertex_count * uv_stride
    else:
        uvs = [(0.0, 0.0)] * vertex_count

    return indices, vertices, uvs, end_pos


def is_degenerate_triangle(vertices: List[Vertex], a: int, b: int, c: int) -> bool:
    """
    Returns True if the triangle has no area (and should be discarded).

    Replaces the original script's `checkit` function, which compared x, y, and z
    at the same time and ended up discarding valid triangles lying on
    axis-aligned planes (e.g., all vertices having the same z).
    """
    if a == b or b == c or a == c:
        return True

    ax, ay, az = vertices[a]
    bx, by, bz = vertices[b]
    cx, cy, cz = vertices[c]

    # Cross product (b-a) x (c-a); the area is half its magnitude
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    return (nx * nx + ny * ny + nz * nz) < 1e-18


def write_obj(
    path: str,
    stem: str,
    mesh_number: int,
    vertices: List[Vertex],
    uvs: List[UV],
    indices: List[int],
) -> None:
    """Writes a mesh in Wavefront OBJ format."""
    lines = [
        "# Wavefront OBJ exported by FBSL Xandreale modeler exporter",
        f"mtllib {stem}.mtl",
        "g dummy",
        f"g Mesh_{mesh_number}",
    ]

    for x, y, z in vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    # Texture coordinates (V is flipped: 1 - v)
    for u, v in uvs:
        lines.append(f"vt {u:.6f} {1.0 - v:.6f}")

    # Faces: triangle list. OBJ indices start at 1.
    for i in range(0, len(indices) - 2, 3):
        a, b, c = indices[i], indices[i + 1], indices[i + 2]
        if max(a, b, c) >= len(vertices):
            continue  # index pointing to a nonexistent vertex
        if not is_degenerate_triangle(vertices, a, b, c):
            lines.append(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {path}")


class MeshGroup(NamedTuple):
    """A mesh group (vertices + indices + UVs) inside a block."""
    vertex_count: int
    face_index_count: int
    index_offset: int   # offsets are already absolute (within the file)
    vertex_offset: int
    uv_offset: int
    has_uvs: bool
    uv_stride: int      # UV bytes per vertex (4 = one pair of half-floats)


def parse_mesh_block(data: bytes, block_start: int, tag: int) -> Optional[List[MeshGroup]]:
    """
    Parses a mesh block header and returns its groups
    (1 group for a simple mesh, 2 for a split mesh).
    Returns None if the block appears invalid (a false-positive signature match).

    Fields used (offsets within the block, all big-endian):
      Simple mesh (288-byte header):
        +24  mvtcount      (uint16) vertex count
        +26  faceidcount   (uint16) index count
        +64  uvarraysize   (uint32) UV array size; / vertices = stride
        +144 offsetindex / +148 offsetvertex / +152 offsetunk / +156 offsetuvs
      Split mesh (560-byte header):
        group A: counts at +24/+26, offsets at +272
        group B: counts at +152/+154, offsets at +416
        UV array size: +64 (group A) and +192 (group B)
      Offsets are relative to the block start; offsetuvs = 0 -> no UVs.
    """
    if tag == MESH_TAG:
        layout = [(24, 144, 64)]
        header_size = 288
    else:
        layout = [(24, 272, 64), (152, 416, 192)]
        header_size = 560

    if block_start + header_size > len(data):
        return None

    groups = []
    for counts_off, offsets_off, uvsize_off in layout:
        v_count, f_count = struct.unpack_from(">HH", data, block_start + counts_off)
        uv_array_size = read_u32(data, block_start + uvsize_off)
        uv_stride = uv_array_size // v_count if v_count else 4
        if uv_stride not in (4, 8):
            uv_stride = 4  # unexpected field: assume the most common layout
        off_index, off_vertex, _off_unk, off_uv = struct.unpack_from(
            ">4I", data, block_start + offsets_off
        )
        group = MeshGroup(
            vertex_count=v_count,
            face_index_count=f_count,
            index_offset=block_start + off_index,
            vertex_offset=block_start + off_vertex,
            uv_offset=block_start + off_uv,
            has_uvs=off_uv > 0,
            uv_stride=uv_stride,
        )

        # Validation: all arrays must fit within the file
        end_index = group.index_offset + f_count * 2
        end_vertex = group.vertex_offset + v_count * 12
        end_uv = group.uv_offset + v_count * uv_stride if group.has_uvs else 0
        if max(end_index, end_vertex, end_uv) > len(data):
            return None
        groups.append(group)

    return groups


def export_meshes(data: bytes, stem: str) -> None:
    """
    Searches the file for mesh blocks and exports each group as .obj.

    Generated filenames:
      simple mesh -> <name>_mesh<N>.obj
      split mesh -> <name>_mesh<N>_1.obj and <name>_mesh<N+1>_2.obj
    """
    # Collect all signatures in file-position order
    blocks = []
    for tag in (MESH_TAG, SPLIT_MESH_TAG):
        for pos in find_all(data, struct.pack(">I", tag)):
            if pos % 4 == 0:  # game blocks are always aligned
                blocks.append((pos, tag))
    blocks.sort()

    mesh_counter = 0
    for block_start, tag in blocks:
        groups = parse_mesh_block(data, block_start, tag)
        if groups is None:
            print(f"[SKIPPED] invalid block at 0x{block_start:X}")
            continue

        for group_number, group in enumerate(groups, start=1):
            mesh_counter += 1
            indices, vertices, uvs, _ = read_mesh_data(
                data,
                group.index_offset,
                group.vertex_offset,
                group.uv_offset,
                group.has_uvs,
                group.face_index_count,
                group.vertex_count,
                group.uv_stride,
            )
            suffix = "" if tag == MESH_TAG else f"_{group_number}"
            write_obj(
                f"{stem}_mesh{mesh_counter}{suffix}.obj",
                stem, mesh_counter, vertices, uvs, indices,
            )


# ----------------------------------------------------------------------------
# TEXTURES
# ----------------------------------------------------------------------------

class TextureInfo(NamedTuple):
    """Texture table entry (36 bytes, big-endian)."""
    tex_id: int       # texture identifier
    offset: int       # data offset, relative to the embedded archive start
    size: int         # data size in bytes
    fmt: int          # format (134/166 = DXT1, 135 = DXT3, 136 = DXT5)
    mip_levels: int   # mipmap count
    unk_b: int        # unknown
    unk_c: int        # unknown
    unk_d: int        # unknown (long)
    width: int        # LARGURA (vem antes da altura na tabela)
    height: int       # altura
    unk_e: int        # unknown (long, sempre 0x10000 nos arquivos testados)
    pitch: int        # bytes per block row (linear textures only; otherwise 0)
    unk_g: int        # unknown (long)


# id, offset, size | fmt, mips, unkb, unkc | unk long | width, height | 3 longs
TEXTURE_ENTRY = struct.Struct(">3I4BI2H3I")  # = 36 bytes


def parse_texture_archive(data: bytes, start: int) -> Optional[List[TextureInfo]]:
    """
    Reads the texture table from an embedded archive starting at `start`.

    Structure (big-endian):
        [signature 0x020101FF][size][count][36-byte-per-texture table]
        ... texture data at the table offsets (relative to `start`) ...

    Returns None if the candidate is invalid (signature found by chance).
    """
    if start + 12 > len(data):
        return None
    count = read_u32(data, start + 8)
    if not 0 < count <= MAX_TEXTURES_PER_ARCHIVE:
        return None
    table_end = start + 12 + count * TEXTURE_ENTRY.size
    if table_end > len(data):
        return None

    infos = []
    for i in range(count):
        info = TextureInfo(*TEXTURE_ENTRY.unpack_from(data, start + 12 + i * TEXTURE_ENTRY.size))
        valid = (
            0 < info.width <= MAX_TEXTURE_DIMENSION
            and 0 < info.height <= MAX_TEXTURE_DIMENSION
            and info.size > 0
            and start + info.offset + info.size <= len(data)
        )
        if not valid:
            return None
        infos.append(info)
    return infos


def find_texture_archives(data: bytes) -> List[Tuple[int, List[TextureInfo]]]:
    """
    Finds all valid embedded texture archives.
    Returns a list of (start_position, [TextureInfo, ...]) in order.
    """
    archives = []
    covered_until = 0  # candidates inside an already parsed archive are false positives
    for start in find_all(data, struct.pack(">I", TEXTURE_ARCHIVE_TAG)):
        if start < covered_until:
            continue
        infos = parse_texture_archive(data, start)
        if infos is None:
            continue
        archives.append((start, infos))
        covered_until = max(start + i.offset + i.size for i in infos)
    return archives


def block_bytes(fourcc: str) -> int:
    """Bytes per 4x4 block: 8 for DXT1, 16 for DXT3/DXT5."""
    return 8 if fourcc == "DXT1" else 16


def mip_chain_size(width: int, height: int, levels: int, block: int) -> int:
    """Total size (in bytes) of a mipmap chain compressed in 4x4 blocks."""
    total = 0
    for level in range(levels):
        w = max(1, width >> level)
        h = max(1, height >> level)
        total += max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block
    return total


def guess_mip_count(width: int, height: int, fourcc: str, data_size: int, hint: int = 0) -> int:
    """
    Determines the number of mipmaps by comparing the data size with the
    expected chain size. Tries `hint` (the table value) first.
    Returns 0 if no count matches.
    """
    block = block_bytes(fourcc)
    candidates = ([hint] if hint > 0 else []) + list(range(1, 16))
    for levels in candidates:
        if mip_chain_size(width, height, levels, block) == data_size:
            return levels
    return 0


# --- Linear textures (format 166): mipmaps stacked with fixed pitch ----

def _level_geometry(width: int, height: int, levels: int):
    """For each mip: (blocks_per_row, block_rows)."""
    for level in range(levels):
        w = max(1, width >> level)
        h = max(1, height >> level)
        yield max(1, (w + 3) // 4), max(1, (h + 3) // 4)


def linear_atlas_to_packed(raw: bytes, info: TextureInfo, block: int) -> Optional[bytes]:
    """
    Converts the linear "atlas" (all rows with the same pitch, mips stacked
    vertically) into the compact DDS mipmap chain.
    Returns None if the geometry does not match the data size.
    """
    geometry = list(_level_geometry(info.width, info.height, info.mip_levels))
    total_rows = sum(rows for _, rows in geometry)
    if info.pitch <= 0 or total_rows * info.pitch != len(raw):
        return None

    out = bytearray()
    row = 0
    for blocks_per_row, rows in geometry:
        for r in range(rows):
            start = (row + r) * info.pitch
            out += raw[start:start + blocks_per_row * block]
        row += rows
    return bytes(out)


def packed_to_linear_atlas(packed: bytes, info: TextureInfo, block: int, original: bytes) -> bytes:
    """
    Inverse operation of `linear_atlas_to_packed`. Starts with the original bytes
    to preserve padding (unused area to the right of smaller mip levels).
    """
    atlas = bytearray(original)
    src = 0
    row = 0
    for blocks_per_row, rows in _level_geometry(info.width, info.height, info.mip_levels):
        length = blocks_per_row * block
        for r in range(rows):
            dst = (row + r) * info.pitch
            atlas[dst:dst + length] = packed[src:src + length]
            src += length
        row += rows
    return bytes(atlas)


def unpack_texture(info: TextureInfo, raw: bytes) -> Tuple[bytes, int]:
    """
    Converts SPK texture data into the standard mipmap chain.
    Returns (packed_data, mipmap_count). Mipmap count = 0 means
    the count could not be identified.
    """
    fourcc = TEXTURE_FORMATS[info.fmt]
    block = block_bytes(fourcc)

    packed = raw
    if info.pitch:
        converted = linear_atlas_to_packed(raw, info, block)
        if converted is not None:
            packed = converted

    levels = guess_mip_count(info.width, info.height, fourcc, len(packed), info.mip_levels)
    return packed, levels


def repack_texture(info: TextureInfo, packed: bytes, original: bytes) -> bytes:
    """Converts the DDS mipmap chain back to the SPK format."""
    if info.pitch and linear_atlas_to_packed(original, info, block_bytes(TEXTURE_FORMATS[info.fmt])) is not None:
        return packed_to_linear_atlas(packed, info, block_bytes(TEXTURE_FORMATS[info.fmt]), original)
    return packed


def build_dds_header(width: int, height: int, fourcc: str, mip_count: int = 0) -> bytes:
    """
    Builds the 128-byte DDS header (4-byte magic + 124-byte header).
    Linear size and mipmap count are filled in so that
    image editors recreate the same mipmap chain when saving.
    """
    linear_size = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * block_bytes(fourcc)

    header = struct.pack(
        "<4s7I44x",
        b"DDS ",      # magic
        124,          # header size
        DDS_FLAGS,    # flags
        height,
        width,
        linear_size,  # level 0 size
        0,            # depth
        mip_count,    # mipmap count (0 = unidentified)
        # 44 bytes reservados (zeros)
    )
    pixel_format = struct.pack(
        "<II4s5I",
        32,                      # tamanho da estrutura pixel format
        DDS_PF_FOURCC,           # flags: format specified by FourCC
        fourcc.encode("ascii"),  # "DXT1", "DXT3" ou "DXT5"
        0, 0, 0, 0, 0,           # bit count and RGBA masks (unused)
    )
    caps = struct.pack("<5I", DDS_CAPS, 0, 0, 0, 0)  # caps1..4 + reserved
    return header + pixel_format + caps  # 128 bytes


def parse_dds(raw: bytes) -> Tuple[int, int, str, bytes]:
    """Reads a .dds and returns (width, height, fourcc, pixel_data)."""
    if raw[:4] != b"DDS " or len(raw) < 128:
        raise ValueError("not a DDS file")
    height, width = struct.unpack_from("<II", raw, 12)
    fourcc = raw[84:88].decode("ascii", "replace")
    if fourcc == "DX10":
        raise ValueError("DDS with a DX10 header; save in legacy format (DXT1/DXT3/DXT5)")
    return width, height, fourcc, raw[128:]



# --- Format 133 (0x85): ARGB8888 uncompressed, swizzled (Morton order) ---

FMT_ARGB8888_SWIZZLED = 133


def _morton_indices(width: int, height: int):
    """Linear pixel position -> index in the PS3 swizzled (Morton) layout."""
    def spread(v: int, bits: int) -> int:
        r = 0
        for i in range(bits):
            r |= ((v >> i) & 1) << (2 * i)
        return r
    bw, bh = width.bit_length() - 1, height.bit_length() - 1
    if (1 << bw) != width or (1 << bh) != height:
        raise ValueError("swizzle requires power-of-two dimensions")
    bits = min(bw, bh)
    out = []
    for y in range(height):
        for x in range(width):
            lo = spread(x & ((1 << bits) - 1), bits) | (spread(y & ((1 << bits) - 1), bits) << 1)
            # the larger dimension continues linearly above the interleaved bits
            hi = ((x >> bits) | (y >> bits)) << (2 * bits)
            out.append(lo | hi)
    return out


def argb_swizzled_to_bgra(raw: bytes, width: int, height: int) -> bytes:
    """PS3 swizzled big-endian ARGB -> linear BGRA bytes (DDS A8R8G8B8)."""
    idx = _morton_indices(width, height)
    out = bytearray(width * height * 4)
    for i, m in enumerate(idx):
        a, r, g, b = raw[m * 4:m * 4 + 4]
        out[i * 4:i * 4 + 4] = bytes((b, g, r, a))
    return bytes(out)


def bgra_to_argb_swizzled(bgra: bytes, width: int, height: int) -> bytes:
    """Inverse of `argb_swizzled_to_bgra` (use this in the injector)."""
    idx = _morton_indices(width, height)
    out = bytearray(width * height * 4)
    for i, m in enumerate(idx):
        b, g, r, a = bgra[i * 4:i * 4 + 4]
        out[m * 4:m * 4 + 4] = bytes((a, r, g, b))
    return bytes(out)


def build_dds_header_argb(width: int, height: int) -> bytes:
    """128-byte DDS header for uncompressed A8R8G8B8 (no mipmaps)."""
    header = struct.pack("<4s7I44x", b"DDS ", 124, 0x1007 | 0x8, height, width,
                         width * 4, 0, 1)
    pixel_format = struct.pack("<8I", 32, 0x41, 0, 32,
                               0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    caps = struct.pack("<5I", 0x1000, 0, 0, 0, 0)
    return header + pixel_format + caps



def parse_dds_argb(raw: bytes) -> Tuple[int, int, bytes]:
    """
    Reads an UNCOMPRESSED .dds (24 or 32 bits, any channel mask order) and
    returns (width, height, bgra_pixels) for the first mip level only.
    """
    if raw[:4] != b"DDS " or len(raw) < 128:
        raise ValueError("not a DDS file")
    height, width = struct.unpack_from("<II", raw, 12)
    pf_flags, fourcc = struct.unpack_from("<I4s", raw, 80)
    bitcount, rmask, gmask, bmask, amask = struct.unpack_from("<5I", raw, 88)
    if pf_flags & 0x4:
        raise ValueError(f"compressed DDS ({fourcc.decode('ascii', 'replace')}); "
                         "this texture needs an uncompressed A8R8G8B8 DDS")
    if bitcount not in (24, 32):
        raise ValueError(f"unsupported DDS bit count: {bitcount}")
    bpp = bitcount // 8
    need = width * height * bpp
    if len(raw) < 128 + need:
        raise ValueError("DDS pixel data is truncated")
    pix = raw[128:128 + need]  # extra mip levels (if any) are ignored

    if bpp == 4 and (rmask, gmask, bmask, amask) == (0xFF0000, 0xFF00, 0xFF, 0xFF000000):
        return width, height, pix  # already BGRA

    def shift(mask):
        return (mask & -mask).bit_length() - 1 if mask else 0

    sr, sg, sb, sa = shift(rmask), shift(gmask), shift(bmask), shift(amask)
    out = bytearray(width * height * 4)
    for i in range(width * height):
        v = int.from_bytes(pix[i * bpp:(i + 1) * bpp], "little")
        out[i * 4] = (v & bmask) >> sb
        out[i * 4 + 1] = (v & gmask) >> sg
        out[i * 4 + 2] = (v & rmask) >> sr
        out[i * 4 + 3] = ((v & amask) >> sa) if amask else 255
    return width, height, bytes(out)


def export_textures(data: bytes, stem: str) -> None:
    """Extracts textures from an "Img.spk"/"Tex.spk" file and saves them as .dds."""
    texture_number = 0  # global counter used in .dds filenames

    for start, infos in find_texture_archives(data):
        print(f"Embedded archive at 0x{start:X}: {len(infos)} texture(s)")
        for info in infos:
            texture_number += 1
            if info.fmt == FMT_ARGB8888_SWIZZLED:
                # Uncompressed texture: saved as PNG (needs Pillow: pip install pillow)
                from PIL import Image
                raw = data[start + info.offset:start + info.offset + info.size]
                bgra = argb_swizzled_to_bgra(raw, info.width, info.height)
                path = f"{stem}_{texture_number}.png"
                Image.frombytes("RGBA", (info.width, info.height), bgra, "raw", "BGRA").save(path)
                print(f"  Written: {path}  (id {info.tex_id}, fmt 133 -> "
                      f"ARGB8888 {info.width}x{info.height}, swizzled, PNG)")
                continue
            if info.fmt not in TEXTURE_FORMATS:
                print(f"  [SKIPPED] #{texture_number}: formato unknown {info.fmt}")
                continue

            fourcc = TEXTURE_FORMATS[info.fmt]
            raw = data[start + info.offset:start + info.offset + info.size]
            packed, levels = unpack_texture(info, raw)
            if levels == 0:
                print(f"  [WARNING] #{texture_number}: mipmaps could not be identified")

            path = f"{stem}_{texture_number}.dds"
            with open(path, "wb") as f:
                f.write(build_dds_header(info.width, info.height, fourcc, levels))
                f.write(packed)
            print(f"  Written: {path}  (id {info.tex_id}, fmt {info.fmt} -> "
                  f"{fourcc} {info.width}x{info.height}, {levels} mips)")


# ----------------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} arquivo.spk")
        sys.exit(1)

    spk_path = sys.argv[1]
    stem = os.path.splitext(os.path.basename(spk_path))[0]

    with open(spk_path, "rb") as f:
        data = f.read()

    # Files ending in "Img.spk" or "Tex.spk" contain textures
    if spk_path.lower().endswith(("img.spk", "tex.spk")):
        export_textures(data, stem)
    else:
        export_meshes(data, stem)


if __name__ == "__main__":
    main()
