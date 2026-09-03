import math

from im2vec.tokenizer import SVGTokenizer, Vocab


def _roundtrip(svg: str) -> str:
    t = SVGTokenizer()
    tokens = t.encode_svg(svg)
    assert tokens[0] == Vocab.SOS
    assert tokens[-1] == Vocab.EOS
    return t.decode_tokens(tokens)


def test_roundtrip_rect():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
        '<rect x="10" y="20" width="100" height="80" fill="#ff0000"/></svg>'
    )
    out = _roundtrip(svg)
    assert 'fill="rgb(255,0,0)"' in out
    # rect -> M L L L Z
    assert out.count("<path") == 1
    d = out.split('d="')[1].split('"')[0]
    assert d.startswith("M 10 20")
    assert d.endswith("Z")


def test_roundtrip_circle_is_cubic():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
        '<circle cx="128" cy="128" r="64" fill="black"/></svg>'
    )
    out = _roundtrip(svg)
    d = out.split('d="')[1].split('"')[0]
    # Arcs are converted to cubic segments (C commands), not A.
    assert " A " not in d
    assert " C " in d


def test_multiple_shapes_and_groups():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
        '<g transform="translate(10 10)">'
        '<path d="M 0 0 L 50 0 L 50 50 Z" fill="#00ff00"/></g>'
        '<polygon points="0,0 20,0 20,20" fill="blue"/></svg>'
    )
    out = _roundtrip(svg)
    assert out.count("<path") == 2
    # translate applied: first point of green path at (10,10)
    assert 'fill="rgb(0,255,0)"' in out
    assert "M 10 10" in out


def test_quadratic_and_smooth_conversion():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
        '<path d="M 10 80 Q 52 10 95 80 T 180 80" fill="none"/></svg>'
    )
    out = _roundtrip(svg)
    d = out.split('d="')[1].split('"')[0]
    assert "Q" not in d
    assert "T" not in d
    assert "C" in d


def test_arc_stays_near_circle():
    # A unit-circle arc should produce cubic control points whose endpoints
    # lie on (or very near) the expected circle.
    from im2vec.tokenizer import _arc_to_cubics

    segs = _arc_to_cubics(128 - 64, 128, 64, 64, 0, True, False, 128 + 64, 128)
    for c1x, c1y, c2x, c2y, x, y in segs:
        for px, py in ((x, y),):
            r = math.hypot(px - 128, py - 128)
            assert abs(r - 64) < 1e-6


def test_vocab_size():
    assert SVGTokenizer().vocab_size == 265


def test_stroke_outline():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
        '<circle cx="128" cy="128" r="64" fill="none" stroke="#0000ff" '
        'stroke-width="4"/></svg>'
    )
    out = _roundtrip(svg)
    assert 'stroke="rgb(0,0,255)"' in out
    assert 'stroke-width="4"' in out
    assert 'fill="none"' in out
    # no filled paths leak in
    assert 'fill="rgb(' not in out
