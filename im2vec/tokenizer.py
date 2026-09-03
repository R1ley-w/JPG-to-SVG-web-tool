"""SVG <-> token sequence.

The network predicts SVG as a sequence of tokens. We normalize arbitrary SVG
input into a small, self-contained subset:

    commands : M L C Z          (absolute coordinates, cubic Bezier)
    color    : flat RGB fill

Token vocabulary (see :class:`Vocab`)::

    0  PAD
    1  SOS
    2  EOS
    3  M
    4  L
    5  C
    6  Z
    7  FILL
    8..            numeric values 0..255 (coordinates and color channels)

A single SVG encodes as::

    [SOS] ( FILL r g b  [path-commands] )* [EOS]

where ``path-commands`` are ``M x y | L x y | C x1 y1 x2 y2 x y | Z``.

Coordinates are normalized into the ``0..255`` range against the SVG viewBox
(the same 256x256 canvas the raster input is rendered on), so a decoded SVG is
pixel-aligned with its training image.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class Vocab:
    """Token vocabulary shared by the tokenizer and the model."""

    PAD = 0
    SOS = 1
    EOS = 2

    M = 3
    L = 4
    C = 5
    Z = 6
    FILL = 7

    NUM_OFFSET = 8  # token id = NUM_OFFSET + value, value in 0..255
    NUM_RANGE = 256

    VOCAB_SIZE = NUM_OFFSET + NUM_RANGE  # 264

    COMMANDS = (M, L, C, Z, FILL)
    COMMAND_NAMES = {
        M: "M",
        L: "L",
        C: "C",
        Z: "Z",
        FILL: "FILL",
    }

    # Arity of each command (number of numeric tokens that follow).
    COMMAND_ARITY = {M: 2, L: 2, C: 6, Z: 0, FILL: 3}


SVG_NS = "http://www.w3.org/2000/svg"


# ---------------------------------------------------------------------------
# Low-level path parsing
# ---------------------------------------------------------------------------

_PATH_TOKEN_RE = re.compile(
    r"([MmLlHhVvCcSsQqTtAaZz])|"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)

_NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}


def _parse_path(d: str) -> List[Tuple[str, List[float]]]:
    """Split a path ``d`` string into (command, [args]) pairs.

    ``command`` is the original (possibly relative) letter. Coordinates are
    kept as floats in their raw form; normalization happens later.
    """
    tokens = _PATH_TOKEN_RE.findall(d)
    commands: List[Tuple[str, List[float]]] = []
    current = None
    args: List[float] = []
    for cmd, num in tokens:
        if cmd:
            if current is not None:
                commands.append((current, args))
            current = cmd
            args = []
        else:
            args.append(float(num))
    if current is not None:
        commands.append((current, args))
    return commands


def _vector_angle(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    """Signed angle (radians) from vector ``u`` to vector ``v``."""
    return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


def _arc_to_cubics(
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    phi_deg: float,
    large_arc: bool,
    sweep: bool,
    x2: float,
    y2: float,
) -> List[Tuple[float, ...]]:
    """Convert an SVG elliptical arc to a list of cubic Bezier segments.

    Each returned tuple is ``(c1x, c1y, c2x, c2y, x, y)``. Follows the SVG 1.1
    implementation notes (F.6.5 center parameterization, F.6.6 segmentation).
    """
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return [(x2, y2, x2, y2, x2, y2)]

    rx, ry = abs(rx), abs(ry)
    phi = math.radians(phi_deg % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    # F.6.5.1 -- translate to origin and rotate into ellipse frame.
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # Ensure radii are large enough.
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s

    rx2, ry2 = rx * rx, ry * ry
    x1p2, y1p2 = x1p * x1p, y1p * y1p

    # F.6.5.2 -- center of the ellipse in the rotated frame.
    num = rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2
    den = rx2 * y1p2 + ry2 * x1p2
    coef = 0.0 if den == 0 else math.sqrt(max(0.0, num / den))
    if large_arc == sweep:
        coef = -coef
    cxp = coef * (rx * y1p) / ry
    cyp = coef * (-ry * x1p) / rx

    # F.6.5.3 -- center in original coordinates.
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    # F.6.5.4 -- start angle and sweep.
    u = ((x1p - cxp) / rx, (y1p - cyp) / ry)
    v = ((-x1p - cxp) / rx, (-y1p - cyp) / ry)
    theta1 = _vector_angle((1.0, 0.0), u)
    delta = _vector_angle(u, v)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    # F.6.6 -- split into <= 90 degree segments and emit cubics.
    n = max(1, int(math.ceil(abs(delta) / (math.pi / 2.0))))
    seg = delta / n
    alpha = 4.0 / 3.0 * math.tan(seg / 4.0)

    cubics: List[Tuple[float, ...]] = []
    a = theta1
    for _ in range(n):
        a1 = a
        a2 = a + seg
        c1_local = (rx * math.cos(a1), ry * math.sin(a1))
        c2_local = (rx * math.cos(a2), ry * math.sin(a2))
        t1 = (-rx * math.sin(a1), ry * math.cos(a1))
        t2 = (-rx * math.sin(a2), ry * math.cos(a2))

        def _rot(x: float, y: float) -> Tuple[float, float]:
            return (cos_phi * x - sin_phi * y + cx, sin_phi * x + cos_phi * y + cy)

        p1 = _rot(c1_local[0] + alpha * t1[0], c1_local[1] + alpha * t1[1])
        p2 = _rot(c2_local[0] - alpha * t2[0], c2_local[1] - alpha * t2[1])
        p3 = _rot(c2_local[0], c2_local[1])
        cubics.append((p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]))
        a = a2
    return cubics


def _abs_cubics(
    commands: Sequence[Tuple[str, List[float]]],
) -> List[Tuple[str, List[float]]]:
    """Convert parsed path commands to absolute ``M/L/C/Z`` (cubic only)."""
    out: List[Tuple[str, List[float]]] = []
    cx = cy = 0.0  # current point
    sx = sy = 0.0  # subpath start
    prev_cmd = ""
    prev_ctrl = (0.0, 0.0)  # last control point (for smooth reflection)

    for cmd, args in commands:
        rel = cmd.islower()
        cmd = cmd.upper()
        n = len(args)

        if cmd == "M":
            i = 0
            while i < n - 1:
                x, y = args[i], args[i + 1]
                if rel:
                    x += cx
                    y += cy
                if i == 0:
                    out.append(("M", [x, y]))
                    sx, sy = x, y
                else:
                    out.append(("L", [x, y]))
                cx, cy = x, y
                i += 2
            prev_ctrl = (cx, cy)
            prev_cmd = "M"

        elif cmd == "L":
            i = 0
            while i < n - 1:
                x, y = args[i], args[i + 1]
                if rel:
                    x += cx
                    y += cy
                out.append(("L", [x, y]))
                cx, cy = x, y
                i += 2
            prev_ctrl = (cx, cy)
            prev_cmd = "L"

        elif cmd == "H":
            i = 0
            while i < n:
                x = args[i] + (cx if rel else 0.0)
                out.append(("L", [x, cy]))
                cx = x
                i += 1
            prev_cmd = "H"

        elif cmd == "V":
            i = 0
            while i < n:
                y = args[i] + (cy if rel else 0.0)
                out.append(("L", [cx, y]))
                cy = y
                i += 1
            prev_cmd = "V"

        elif cmd == "C":
            i = 0
            while i + 5 < n:
                c1x, c1y, c2x, c2y, x, y = args[i : i + 6]
                if rel:
                    c1x += cx
                    c1y += cy
                    c2x += cx
                    c2y += cy
                    x += cx
                    y += cy
                out.append(("C", [c1x, c1y, c2x, c2y, x, y]))
                prev_ctrl = (c2x, c2y)
                cx, cy = x, y
                i += 6
            prev_cmd = "C"

        elif cmd == "S":
            i = 0
            while i + 3 < n:
                c2x, c2y, x, y = args[i : i + 4]
                if rel:
                    c2x += cx
                    c2y += cy
                    x += cx
                    y += cy
                if prev_cmd in ("C", "S"):
                    c1x = 2 * cx - prev_ctrl[0]
                    c1y = 2 * cy - prev_ctrl[1]
                else:
                    c1x, c1y = cx, cy
                out.append(("C", [c1x, c1y, c2x, c2y, x, y]))
                prev_ctrl = (c2x, c2y)
                cx, cy = x, y
                i += 4
            prev_cmd = "S"

        elif cmd == "Q":
            i = 0
            while i + 3 < n:
                qx, qy, x, y = args[i : i + 4]
                if rel:
                    qx += cx
                    qy += cy
                    x += cx
                    y += cy
                c1x = cx + 2.0 / 3.0 * (qx - cx)
                c1y = cy + 2.0 / 3.0 * (qy - cy)
                c2x = x + 2.0 / 3.0 * (qx - x)
                c2y = y + 2.0 / 3.0 * (qy - y)
                out.append(("C", [c1x, c1y, c2x, c2y, x, y]))
                prev_ctrl = (qx, qy)
                cx, cy = x, y
                i += 4
            prev_cmd = "Q"

        elif cmd == "T":
            i = 0
            while i + 1 < n:
                x, y = args[i], args[i + 1]
                if rel:
                    x += cx
                    y += cy
                if prev_cmd in ("Q", "T"):
                    qx = 2 * cx - prev_ctrl[0]
                    qy = 2 * cy - prev_ctrl[1]
                else:
                    qx, qy = cx, cy
                c1x = cx + 2.0 / 3.0 * (qx - cx)
                c1y = cy + 2.0 / 3.0 * (qy - cy)
                c2x = x + 2.0 / 3.0 * (qx - x)
                c2y = y + 2.0 / 3.0 * (qy - y)
                out.append(("C", [c1x, c1y, c2x, c2y, x, y]))
                prev_ctrl = (qx, qy)
                cx, cy = x, y
                i += 2
            prev_cmd = "T"

        elif cmd == "A":
            i = 0
            while i + 6 < n:
                rx, ry, rot, laf, sf, x, y = args[i : i + 7]
                if rel:
                    x += cx
                    y += cy
                for seg in _arc_to_cubics(cx, cy, rx, ry, rot, bool(laf), bool(sf), x, y):
                    out.append(("C", list(seg)))
                cx, cy = x, y
                i += 7
            prev_cmd = "A"

        elif cmd == "Z":
            out.append(("Z", []))
            cx, cy = sx, sy
            prev_cmd = "Z"

    return out


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def _parse_transform(spec: str) -> Tuple[float, ...]:
    """Parse an SVG ``transform`` attribute into a 6-tuple (a,b,c,d,e,f)."""
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not spec:
        return m

    for name, argstr in re.findall(r"(\w+)\s*\(([^)]*)\)", spec):
        vals = [float(v) for v in argstr.replace(",", " ").split()]
        if name == "matrix" and len(vals) == 6:
            t = tuple(vals)
        elif name == "translate":
            tx = vals[0]
            ty = vals[1] if len(vals) > 1 else 0.0
            t = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = vals[0]
            sy = vals[1] if len(vals) > 1 else sx
            t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate":
            ang = math.radians(vals[0])
            ca, sa = math.cos(ang), math.sin(ang)
            if len(vals) == 3:
                cx, cy = vals[1], vals[2]
                t = (
                    ca,
                    sa,
                    -sa,
                    ca,
                    cx - ca * cx + sa * cy,
                    cy - sa * cx - ca * cy,
                )
            else:
                t = (ca, sa, -sa, ca, 0.0, 0.0)
        else:
            continue
        m = _compose(m, t)
    return m


def _compose(
    m: Tuple[float, ...], t: Tuple[float, ...]
) -> Tuple[float, ...]:
    """Compose ``m`` then ``t`` (both (a,b,c,d,e,f) matrices)."""
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = t
    return (
        a2 * a + c2 * b,
        b2 * a + d2 * b,
        a2 * c + c2 * d,
        b2 * c + d2 * d,
        a2 * e + c2 * f + e2,
        b2 * e + d2 * f + f2,
    )


def _apply_matrix(
    m: Tuple[float, ...], x: float, y: float
) -> Tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


# ---------------------------------------------------------------------------
# Colors and shape conversion
# ---------------------------------------------------------------------------


def _parse_color(value: Optional[str]) -> Tuple[int, int, int]:
    """Parse an SVG color into RGB ints. Defaults to black when unknown."""
    if value is None:
        return (0, 0, 0)
    value = value.strip().lower()
    if value in ("none", "transparent"):
        return (0, 0, 0)
    if value in _NAMED_COLORS:
        return _NAMED_COLORS[value]
    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return (0, 0, 0)
    m = re.match(r"rgba?\(([^)]*)\)", value)
    if m:
        parts = [p.strip().rstrip("%") for p in m.group(1).split(",")]
        if len(parts) >= 3:
            try:
                rgb = [int(float(p)) for p in parts[:3]]
                rgb = [max(0, min(255, c)) for c in rgb]
                return (rgb[0], rgb[1], rgb[2])
            except ValueError:
                return (0, 0, 0)
    return (0, 0, 0)


def _shape_to_path_d(elem: ET.Element) -> Optional[str]:
    """Convert a basic SVG shape element into an equivalent path ``d`` string."""
    tag = elem.tag.split("}")[-1].lower()
    try:
        if tag == "rect":
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
            return f"M {x} {y} L {x + w} {y} L {x + w} {y + h} L {x} {y + h} Z"
        if tag == "circle":
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            r = float(elem.get("r", 0))
            if r <= 0:
                return None
            return (
                f"M {cx - r} {cy} A {r} {r} 0 1 0 {cx + r} {cy} "
                f"A {r} {r} 0 1 0 {cx - r} {cy} Z"
            )
        if tag == "ellipse":
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            rx = float(elem.get("rx", 0))
            ry = float(elem.get("ry", 0))
            if rx <= 0 or ry <= 0:
                return None
            return (
                f"M {cx - rx} {cy} A {rx} {ry} 0 1 0 {cx + rx} {cy} "
                f"A {rx} {ry} 0 1 0 {cx - rx} {cy} Z"
            )
        if tag in ("polygon", "polyline"):
            pts = [float(v) for v in elem.get("points", "").replace(",", " ").split()]
            if len(pts) < 4:
                return None
            d = "M " + " ".join(str(v) for v in pts[:2])
            for i in range(2, len(pts) - 1, 2):
                d += f" L {pts[i]} {pts[i + 1]}"
            d += " Z"
            return d
    except ValueError:
        return None
    return None


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

CANVAS_SIZE = 256  # normalized coordinate range is 0..(CANVAS_SIZE - 1)


class SVGTokenizer:
    """Encodes SVGs to token sequences and decodes them back to SVG."""

    vocab = Vocab()

    def __init__(self, canvas_size: int = CANVAS_SIZE):
        self.canvas_size = canvas_size

    # -- encoding ---------------------------------------------------------

    def encode_svg(self, svg_text: str) -> List[int]:
        """Encode an SVG document into a token sequence (with SOS/EOS)."""
        shapes = self._extract_shapes(svg_text)
        tokens = [Vocab.SOS]
        for d, fill in shapes:
            tokens.append(Vocab.FILL)
            tokens.extend(Vocab.NUM_OFFSET + c for c in fill)
            tokens.extend(self._encode_path(d))
        tokens.append(Vocab.EOS)
        return tokens

    def _extract_shapes(
        self, svg_text: str
    ) -> List[Tuple[str, Tuple[int, int, int]]]:
        """Return a list of (path ``d``, fill) for every drawable element."""
        root = ET.fromstring(svg_text)
        viewbox = self._viewbox(root)
        shapes: List[Tuple[str, Tuple[int, int, int]]] = []

        def walk(elem: ET.Element, matrix: Tuple[float, ...]) -> None:
            local = _compose(matrix, _parse_transform(elem.get("transform")))
            tag = elem.tag.split("}")[-1].lower()
            fill = _parse_color(elem.get("fill"))
            if tag == "path":
                d = elem.get("d")
                if d:
                    shapes.append((self._normalize(_abs_cubics(_parse_path(d)), local, viewbox), fill))
            else:
                d = _shape_to_path_d(elem)
                if d is not None:
                    shapes.append((self._normalize(_abs_cubics(_parse_path(d)), local, viewbox), fill))
            for child in elem:
                walk(child, local)

        walk(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        return shapes

    def _viewbox(self, root: ET.Element) -> Tuple[float, float, float, float]:
        vb = root.get("viewBox")
        if vb:
            parts = [float(v) for v in vb.replace(",", " ").split()]
            if len(parts) == 4:
                return (parts[0], parts[1], parts[2], parts[3])
        w = float(root.get("width", self.canvas_size) or self.canvas_size)
        h = float(root.get("height", self.canvas_size) or self.canvas_size)
        return (0.0, 0.0, w, h)

    def _normalize(
        self,
        commands: List[Tuple[str, List[float]]],
        matrix: Tuple[float, ...],
        viewbox: Tuple[float, float, float, float],
    ) -> str:
        minx, miny, vw, vh = viewbox
        sx = (self.canvas_size - 1) / vw if vw else 1.0
        sy = (self.canvas_size - 1) / vh if vh else 1.0
        parts: List[str] = []
        for cmd, args in commands:
            if cmd == "Z":
                parts.append("Z")
                continue
            nums: List[int] = []
            for i in range(0, len(args), 2):
                x, y = _apply_matrix(matrix, args[i], args[i + 1])
                nx = round((x - minx) * sx)
                ny = round((y - miny) * sy)
                nx = max(0, min(self.canvas_size - 1, int(nx)))
                ny = max(0, min(self.canvas_size - 1, int(ny)))
                nums.extend((nx, ny))
            parts.append(cmd + " " + " ".join(str(v) for v in nums))
        return " ".join(parts)

    def _encode_path(self, d: str) -> List[int]:
        """Encode a normalized path ``d`` string into command + number tokens."""
        out: List[int] = []
        for cmd, argstr in re.findall(r"([MLCZ])|(-?\d+)", d):
            if cmd:
                out.append(getattr(Vocab, cmd))
            elif argstr:
                out.append(Vocab.NUM_OFFSET + int(argstr))
        return out

    # -- decoding ---------------------------------------------------------

    def decode_tokens(self, tokens: Sequence[int]) -> str:
        """Decode a token sequence back into an SVG document string."""
        n = len(tokens)
        shapes: List[str] = []
        i = 0
        if i < n and tokens[i] == Vocab.SOS:
            i += 1
        while i < n:
            tok = tokens[i]
            if tok in (Vocab.EOS, Vocab.PAD):
                break
            if tok == Vocab.FILL:
                r = tokens[i + 1] - Vocab.NUM_OFFSET
                g = tokens[i + 2] - Vocab.NUM_OFFSET
                b = tokens[i + 3] - Vocab.NUM_OFFSET
                i += 4
                d, i = self._decode_path(tokens, i)
                shapes.append(f'<path fill="rgb({r},{g},{b})" d="{d}"/>')
            else:
                i += 1

        body = "\n".join("  " + s for s in shapes)
        return (
            f'<svg xmlns="{SVG_NS}" viewBox="0 0 {self.canvas_size} '
            f'{self.canvas_size}" width="{self.canvas_size}" '
            f'height="{self.canvas_size}">\n{body}\n</svg>'
        )

    def _decode_path(self, tokens: Sequence[int], i: int) -> Tuple[str, int]:
        parts: List[str] = []
        n = len(tokens)
        while i < n:
            tok = tokens[i]
            if tok in (Vocab.EOS, Vocab.PAD, Vocab.FILL):
                break
            cmd = Vocab.COMMAND_NAMES.get(tok)
            if cmd is None:
                i += 1
                continue
            arity = Vocab.COMMAND_ARITY[tok]
            nums = [
                tokens[j] - Vocab.NUM_OFFSET
                for j in range(i + 1, min(i + 1 + arity, n))
            ]
            if arity:
                parts.append(cmd + " " + " ".join(str(v) for v in nums))
            else:
                parts.append(cmd)
            i += 1 + arity
        return " ".join(parts), i

    @property
    def vocab_size(self) -> int:
        return Vocab.VOCAB_SIZE
