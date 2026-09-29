"""Optional presentation-only XIC captions; no source pixels or measurements edited.

All caption choices are off by default. Provenance remains in the workbook audit
and DrawingML alternative description, independently of visible caption choices.
No templates/eval or user data are needed for custom text: it is literal text.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import unicodedata

MAX_CUSTOM_LENGTH = 200


def clean_text(value):
    text = str(value if value is not None else '')
    # Remove XML-forbidden controls; collapse whitespace for a compact header.
    return ' '.join(''.join(c for c in text if not unicodedata.category(c).startswith('C')
                           or c in '\n\r\t').split())


@dataclass(frozen=True)
class CaptionSettings:
    show_replicate: bool = False
    show_raw: bool = False
    show_name: bool = False
    custom_text: str = ''

    def validate(self):
        for flag in (self.show_replicate, self.show_raw, self.show_name):
            if not isinstance(flag, bool):
                raise ValueError('Caption options must be Boolean values.')
        if not isinstance(self.custom_text, str):
            raise ValueError('Custom captions must be text.')
        if len(self.custom_text) > MAX_CUSTOM_LENGTH:
            raise ValueError('Custom captions are limited to 200 characters.')
        return self

    @property
    def enabled(self):
        return bool(self.show_replicate or self.show_raw or self.show_name or clean_text(self.custom_text))

    def field_names(self):
        return [name for name, flag in (('replicate', self.show_replicate),
                ('raw', self.show_raw), ('compound_name', self.show_name),
                ('custom_text', bool(clean_text(self.custom_text)))) if flag]

    def description(self):
        labels = {'replicate': 'Replicate number', 'raw': 'RAW filename',
                  'compound_name': 'Compound name in this table', 'custom_text': 'Custom caption'}
        return ' + '.join(labels[x] for x in self.field_names()) or 'No additional captions (default)'


def from_options(options):
    return CaptionSettings(getattr(options, 'xic_caption_replicate', False),
                           getattr(options, 'xic_caption_raw', False),
                           getattr(options, 'xic_caption_name', False),
                           getattr(options, 'xic_caption_custom', '')).validate()


def build_caption(entry, settings=None, compound_name=''):
    settings = (settings or CaptionSettings()).validate()
    parts = []
    rep = clean_text(entry.get('rep', ''))
    raw = clean_text(entry.get('raw', ''))
    name = clean_text(compound_name)
    if settings.show_replicate and rep in ('1', '2', '3'):
        parts.append('Rep ' + rep)
    # Do not infer a RAW or replicate identity from an arbitrary picture name.
    if settings.show_raw and raw:
        parts.append(raw)
    if settings.show_name and name:
        parts.append(name)
    if clean_text(settings.custom_text):
        parts.append(clean_text(settings.custom_text))
    return ' | '.join(parts)


@lru_cache(maxsize=6)
def load_caption_font(size=20):
    from PIL import ImageFont
    for name in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
                 '/System/Library/Fonts/PingFang.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 'C:/Windows/Fonts/segoeui.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(name).is_file():
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def fit_text(draw, text, font, width, max_lines=2):
    """Bounded line wrapping; full unabridged caption is retained in the audit."""
    remaining, lines = clean_text(text), []
    for index in range(max_lines):
        if not remaining:
            break
        if _width(draw, remaining, font) <= width:
            lines.append(remaining)
            break
        suffix = '...' if index == max_lines - 1 else ''
        lo, hi = 0, len(remaining)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _width(draw, remaining[:mid] + suffix, font) <= width:
                lo = mid
            else:
                hi = mid - 1
        if lo <= 0:
            lines.append('...')
            break
        cut = lo
        if not suffix:
            space = remaining.rfind(' ', 0, lo + 1)
            if space >= lo // 2:
                cut = max(1, space)
        lines.append(remaining[:cut].rstrip() + suffix)
        remaining = remaining[cut:].lstrip()
    return lines


def draw_caption(canvas, bounds, entry, settings=None, compound_name=''):
    """Draw only requested text above an image. Returns image box and audit detail."""
    from PIL import ImageDraw
    x0, y0, x1, y1 = bounds
    draw = ImageDraw.Draw(canvas)
    caption = build_caption(entry, settings, compound_name)
    header = 0
    visible = ''
    if caption:
        font = load_caption_font(20)
        lines = fit_text(draw, caption, font, max(20, x1-x0-20),
                         max_lines=2 if y1-y0 >= 150 else 1)
        line_height = max(24, font.getbbox('Ag')[3] - font.getbbox('Ag')[1] + 4)
        header = len(lines) * line_height + 12
        for i, line in enumerate(lines):
            # Anchor at top with bbox offset so accented/CJK glyphs are not clipped.
            box = draw.textbbox((0, 0), line, font=font)
            draw.text((x0+10, y0+6+i*line_height-box[1]), line, font=font, fill='black')
        visible = '\n'.join(lines)
    area = (x0+10, y0+max(6, header), x1-10, y1-8)
    detail = {'rep': str(entry.get('rep', '')), 'full_caption': caption,
              'visible_caption': visible, 'caption_truncated': '...' in visible and visible != caption,
              'header_pixels': header}
    return area, detail


def paste_original(canvas, entry, area):
    """Only EXIF orientation, scale and alpha compositing; never crop original data."""
    from PIL import Image, ImageOps
    x0, y0, x1, y1 = area
    with Image.open(entry['path']) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGBA')
        image.thumbnail((max(1, x1-x0), max(1, y1-y0)),
                        getattr(Image, 'Resampling', Image).LANCZOS)
        pos = (x0 + (x1-x0-image.width)//2, y0 + (y1-y0-image.height)//2)
        canvas.paste(image, pos, image)


def draw_placeholder(canvas, area, text):
    # A missing/conflicting slot must not masquerade as a measured zero/flat XIC.
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    font = load_caption_font(20)
    x0, y0, x1, y1 = area
    lines = fit_text(draw, text, font, max(20, x1-x0), max_lines=2)
    for i, line in enumerate(lines):
        draw.text((x0+5, y0+max(0, (y1-y0-len(lines)*25)//2)+i*25),
                  line, font=font, fill='black')


def run_caption_selftest():
    """Frozen-build smoke check of opt-in pixel overlays; only synthetic images."""
    import copy
    import io
    import tempfile
    from PIL import Image, ImageChops
    from .final_table_display import render_xic
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        entries = []
        for rep in '123':
            path = root / ('SYNTHETIC_' + rep + '.png')
            Image.new('RGB', (400, 100), 'white').save(path)
            entries.append({'path': path, 'rep': rep, 'raw': 'SYNTHETIC-' + rep})
        original = {'paths': [x['path'] for x in entries], 'selected': entries,
                    'conflicts': {}, 'is_triplicate': True}
        no_text = copy.deepcopy(original)
        picture = render_xic(no_text, 520, 220, 'first')
        image = Image.open(io.BytesIO(picture[0])).convert('RGB')
        if ImageChops.difference(image, Image.new('RGB', image.size, 'white')).getbbox():
            raise RuntimeError('Default XIC unexpectedly adds visible text')
        on = copy.deepcopy(original)
        render_xic(on, 520, 480, 'all', CaptionSettings(True, False, True, 'Review'),
                   compound_name='SYNTHETIC')
        if len(on['used']) != 3:
            raise RuntimeError('Caption rendering changed replicate selection')
        if [x['full_caption'] for x in on['caption_records']] != [
                'Rep ' + r + ' | SYNTHETIC | Review' for r in '123']:
            raise RuntimeError('XIC selective caption fields incorrect')
        if from_options(type('Default', (), {})()).enabled:
            raise RuntimeError('Caption defaults must be off')
    return 'default no-overlay pixels; opt-in replicate/name/custom labels; all three sources kept; no measurements used'
