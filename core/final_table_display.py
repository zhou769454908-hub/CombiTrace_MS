"""Presentation-only XIC selection after identity/replicate conflicts are resolved.

Matching remains strict. ``first`` never means take the first ambiguous file:
only entries already accepted by the matcher/claim checks may be displayed.
"""
from __future__ import annotations

import io
from pathlib import Path

DISPLAY_LABELS = {
    'first': 'First available image (prefer Rep 1)',
    'all': 'All three replicates',
}


def validate_display_mode(mode):
    if mode not in DISPLAY_LABELS:
        raise ValueError('Select either the first available image or all three replicates.')
    return mode


def ordered_entries(result):
    """Stable replicate order, independent of filesystem enumeration order."""
    return sorted(result.get('selected', []), key=lambda x: (
        {'1': 0, '2': 1, '3': 2}.get(str(x.get('rep', '')), 3),
        str(x.get('path', '')).casefold(), str(x.get('path', ''))))


def display_preview(result, mode):
    """No pixel reads; export can fall back if an accepted image is unreadable."""
    validate_display_mode(mode)
    entries = ordered_entries(result)
    if not entries:
        return 'No image can be displayed safely'
    if mode == 'first':
        rep = entries[0].get('rep', '')
        return ('Planned display: Rep ' + str(rep) if rep else 'Planned display: single image') + ' (readability checked during export)'
    if result.get('is_triplicate'):
        return 'Triplicate layout; %s/3 images matched' % len(entries)
    return 'Single image (no triplicate identity)'


def _labelled_single(entry, width, height, captions=None, compound_name='', result=None):
    """Embed the full original image; an optional header takes space only if selected."""
    from PIL import Image
    from .final_table_captions import draw_caption, paste_original
    W, H = width * 2, height * 2
    canvas = Image.new('RGB', (W, H), 'white')
    area, detail = draw_caption(canvas, (0, 0, W, H), entry, captions, compound_name)
    paste_original(canvas, entry, area)
    if result is not None:
        result['caption_records'].append(detail)
    out = io.BytesIO(); canvas.save(out, format='PNG', optimize=True)
    # Alternative description preserves source provenance, not visible overlay text.
    rep = str(entry.get('rep', ''))
    label = ('Rep ' + rep if rep else 'Single XIC')
    if entry.get('raw'):
        label += ' | ' + str(entry['raw'])
    return out.getvalue(), width, height, label + ' | ' + str(entry['path'])


def render_xic(result, width, height, mode, captions=None, *, compound_name=''):
    """Render requested layout, recording only actually used sources in ``used``.

    A missing/conflicting/unreadable Rep 1 can fall back to accepted Rep 2, then
    Rep 3. No quality-based/peak-size selection, averaging, or re-integration.
    Matching candidates/selected entries remain intact for provenance auditing.
    """
    validate_display_mode(mode)
    from .final_table_captions import CaptionSettings
    captions = (captions or CaptionSettings()).validate()
    result['caption_records'] = []
    result['used'] = []
    result['read_errors'] = []
    result['display_note'] = ''
    if not result.get('paths'):
        raise ValueError('No safely matched XIC image')
    if mode == 'all' and result.get('is_triplicate'):
        from .final_table_triplicates import compose_triplicates
        picture = compose_triplicates(result, width, height, captions=captions, compound_name=compound_name)
        used = result['used']
        missing = [rep for rep in '123' if rep not in {str(x['rep']) for x in used}]
        result['status'] = 'EMBEDDED_REPLICATES_%s/3' % len(used)
        if missing:
            result['status'] += '; MISSING_OR_CONFLICT_REP' + ','.join(missing)
        result['display_note'] = 'All-replicate layout; original images only, not an averaged trace.'
        return picture
    entries = ordered_entries(result)
    # Non-triplicate matches are unique under the existing strict matching rules.
    for entry in entries:
        try:
            if (mode == 'first' and entry.get('rep')) or captions.enabled:
                picture = _labelled_single(entry, width, height, captions, compound_name, result)
            else:
                from .final_table_postprocess import thumbnail
                picture = thumbnail(entry['path'], width, height)
                result['caption_records'].append({'rep': str(entry.get('rep', '')),
                    'full_caption': '', 'visible_caption': '', 'caption_truncated': False, 'header_pixels': 0})
        except Exception as exc:
            result['read_errors'].append((str(entry.get('rep', '')), str(entry['path']),
                                          type(exc).__name__ + ': ' + str(exc)))
            continue
        result['used'] = [entry]
        rep = str(entry.get('rep', ''))
        result['status'] = 'EMBEDDED_FIRST_REP' + rep if mode == 'first' and rep else 'EMBEDDED'
        reasons = []
        if mode == 'first' and rep in '123' and rep:
            for earlier in '123'[:'123'.index(rep)]:
                if earlier in result.get('conflicts', {}):
                    reasons.append('REP' + earlier + '_CONFLICT_NOT_SELECTED')
                elif any(x[0] == earlier for x in result['read_errors']):
                    reasons.append('REP' + earlier + '_READ_FAILED')
                else:
                    reasons.append('REP' + earlier + '_NOT_AVAILABLE')
        if reasons:
            result['status'] += '; ' + '; '.join(reasons)
        result['display_note'] = (
            'First readable unambiguous image in Rep1/Rep2/Rep3 order; remaining matched images '
            'are intentionally not displayed. No selection by peak size or QC outcome.'
            if mode == 'first' else 'Single matched image; no replicate group identified.')
        return picture
    raise ValueError('No readable XIC image among the unambiguous matches')
