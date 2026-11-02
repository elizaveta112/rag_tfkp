"""Deterministic, font/geometry-aware extraction for the supplied TeX PDF.

No QA/gold input and no generative model. Heuristics are exposed in the audit.
"""
import re
from collections import Counter
from dataclasses import dataclass, replace
from formulas import UNICODE

CMEX = {0: '(', 1: ')', 2: '[', 3: ']', 8: r'\{', 9: r'\}',
        16: '(', 17: ')', 18: '(', 19: ')', 20: '[', 21: ']',
        26: r'\{', 27: r'\}', 40: r'\{', 41: r'\}',
        72: r'\oint ', 73: r'\oint ', 80: r'\sum ', 81: r'\prod ',
        82: r'\int ', 83: r'\bigcup ', 84: r'\bigcap ', 85: r'\biguplus ',
        88: r'\sum ', 89: r'\prod ', 90: r'\int ', 91: r'\bigcup ',
        92: r'\bigcap ', 93: r'\biguplus ', 94: r'\bigwedge ', 95: r'\bigvee ',
        97: r'\coprod ', 98: r'\widehat ', 101: r'\widetilde ',
        112: r'\sqrt ', 114: r'\sqrt ', 115: r'\sqrt ',
        0xf8eb: '(', 0xf8ec: '(', 0xf8ed: '(',
        0xf8f6: ')', 0xf8f7: ')', 0xf8f8: ')',
        0xf8ee: '[', 0xf8ef: '[', 0xf8f0: '[',
        0xf8f9: ']', 0xf8fa: ']', 0xf8fb: ']',
        0xf8f1: r'\{', 0xf8f2: r'\{', 0xf8f3: r'\{',
        0xf8fc: r'\}', 0xf8fd: r'\}', 0xf8fe: r'\}', 0xf8f4: '|'}


@dataclass(eq=False)
class Node:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    base: float
    size: float
    math: bool
    font: str = ''
    display_limits: bool = False

    @property
    def cx(self):
        return (self.x0+self.x1)/2


def decode_text(text):
    # The PDF's Type3 Cyrillic bytes are mapped to Latin-1 codepoints.
    if text == '\x9f':
        return '§'
    try:
        text = text.encode('latin1').decode('cp1251')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text.replace('\x15', '-').replace('\x16', '—').replace('\x1c', '«').replace('\x1d', '»')


def symbol(char, font, audit):
    if font.startswith('CMEX'):
        value = CMEX.get(ord(char))
        if value is None:
            audit['unknown_glyphs'][f'{font}:U+{ord(char):04X}'] += 1
            return r'\pdfglyph{'+f'{font}:{ord(char):04X}'+'}'
        return value
    if char == '′':
        return "'"
    if char == '\\':
        return r'\setminus '
    if char in '{}':
        return '\\'+char
    if char in ('∆', '⩽', '⩾', '·'):
        return {'∆': r'\Delta ', '⩽': r'\le ', '⩾': r'\ge ', '·': r'\cdot '}[char]
    if char == '\u0338':
        return r'\not '
    if font.startswith('MSBM') and char.isalpha():
        return r'\mathbb{'+char+'}'
    if font.startswith('CMSY') and re.fullmatch('[A-Z]', char):
        return r'\mathcal{'+char+'}'
    if ord(char) < 32:
        audit['unknown_glyphs'][f'{font}:U+{ord(char):04X}'] += 1
        return r'\pdfglyph{'+f'{font}:{ord(char):04X}'+'}'
    return UNICODE.get(char, char)


def raw_nodes(page, audit):
    nodes = []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                font, size = span['font'], span['size']
                math = not re.fullmatch(r'F\d+', font)
                for c in span['chars']:
                    if c['c'].isspace() or c['origin'][1] < 44 or c['origin'][1] > page.rect.height-16:
                        continue
                    text = symbol(c['c'], font, audit) if math else decode_text(c['c'])
                    x0, y0, x1, y1 = c['bbox']
                    baseline = c['origin'][1]
                    # Large CMEX delimiters have a top-origin TeX metric.
                    if font.startswith('CMEX'):
                        if c['c'] in ('\x12', '\x13'):
                            baseline += 1.41*size
                        elif c['c'] in ('\x10', '\x11'):
                            baseline += 0.91*size
                        elif text in (r'\int ', r'\oint '):
                            baseline += (1.361 if ord(c['c']) in (90, 73) else .64)*size
                        elif text in (r'\sum ', r'\prod '):
                            baseline += (.75 if ord(c['c']) in (80, 81) else .95)*size
                    display_limits = font.startswith('CMEX') and ord(c['c']) in (88, 89, 90, 73)
                    nodes.append(Node(text, x0, y0, x1, y1, baseline, size, math, font, display_limits))
    return nodes


def horizontal_rules(page):
    result = []
    for drawing in page.get_drawings():
        if (drawing.get('width') or 0) > 1:
            continue
        for item in drawing['items']:
            if item[0] == 'l':
                a, b = item[1:]
                x0, x1 = sorted((a.x, b.x))
                if abs(a.y-b.y) < .15 and 2 <= x1-x0 <= 210 and 44 < a.y < page.rect.height-16:
                    result.append((x0, x1, a.y))
    return result


def region(nodes, x0, x1):
    return [n for n in nodes if n.math and x0-.45 <= n.cx <= x1+.45]


def math_sequence(nodes):
    if not nodes:
        return ''
    # Primary line = the baseline carrying the most full-size ink.
    weights = Counter()
    largest = max(n.size for n in nodes)
    # Three tiny exponent glyphs can carry more ink than their one-letter base.
    # Pick the body baseline from full-size glyphs, not the total script width.
    for n in nodes:
        if n.size < largest*.9:
            continue
        weights[round(n.base*1.25)/1.25] += n.size*(n.x1-n.x0+.2)
    key = weights.most_common(1)[0][0]
    base = min(nodes, key=lambda n: abs(n.base-key)).base
    normal_size = max((n.size for n in nodes if abs(n.base-base) < 1.5), default=10)
    main = [n for n in nodes if not (n.size < normal_size*.87 and abs(n.base-base) > normal_size*.10)]
    scripts = [n for n in nodes if n not in main]
    attachments = {}
    for n in scripts:
        candidates = [m for m in main if m.x0 < n.x0+1.5 and n.x0-m.x1 < normal_size*2]
        if not candidates:
            main.append(n)
            continue
        parent = min(candidates, key=lambda m: abs(n.x0-m.x1))
        attachments.setdefault(parent, {'up': [], 'down': []})['up' if n.base < base else 'down'].append(n)
    parts = []
    for n in sorted(main, key=lambda m: (m.x0, m.base)):
        text = n.text
        for side, marker in (('up', '^'), ('down', '_')):
            group = attachments.get(n, {}).get(side, [])
            if group:
                raw = ''.join(m.text for m in sorted(group, key=lambda m: m.x0))
                text += raw if side == 'up' and set(raw) == {"'"} else marker+'{'+raw+'}'
        parts.append(text)
    return ''.join(parts)


def restore_radicals(nodes, rules, audit):
    """Use the root glyph plus its own horizontal rule; do not infer a root."""
    consumed = set()
    for root in [n for n in nodes if n.text.strip() in (r'\sqrt', '√')]:
        matches = [r for r in rules if abs(r[0]-root.x1) < 1 and abs(r[2]-root.base) < 1]
        if len(matches) != 1:
            audit['unresolved_radicals'] = audit.get('unresolved_radicals', 0)+1
            continue
        x0, x1, y = matches[0]
        inside = [n for n in nodes if n is not root and n.math and
                  x0-.4 <= n.cx <= x1+.4 and .15*root.size < n.base-y < 1.7*root.size]
        if not inside:
            continue
        degree = [n for n in nodes if n is not root and n not in inside and n.math and
                  n.size < .75*root.size and root.x0-1 <= n.cx < x0 and
                  .1*root.size < n.base-y < .75*root.size]
        size = max(n.size for n in inside)
        base = min((n for n in inside if n.size >= .9*size), key=lambda n: n.x0).base
        text = r'\sqrt'+('['+math_sequence(degree)+']' if degree else '')+'{'+math_sequence(inside)+'}'
        new = Node(text, root.x0, min(n.y0 for n in [root]+inside), x1,
                   max(n.y1 for n in inside), base, size, True, 'radical')
        nodes = [n for n in nodes if n not in [root]+inside+degree]+[new]
        consumed.add(matches[0])
        audit['radicals'] = audit.get('radicals', 0)+1
    return nodes, [r for r in rules if r not in consumed]


def restore_geometry(nodes, rules, audit):
    nodes, rules = restore_radicals(nodes, rules, audit)
    fractions = []
    # Identify fractions first, using numerator/denominator baseline geometry.
    for x0, x1, y in rules:
        column = region(nodes, x0, x1)
        above = [n for n in column if .15*n.size < y-n.base < .7*n.size]
        below = [n for n in column if .72*n.size < n.base-y < 1.7*n.size]
        if above and below:
            fractions.append((x0, x1, y))
    # The remaining short rules can be overbars. A bar is NOT inferred from text.
    for x0, x1, y in rules:
        if (x0, x1, y) in fractions:
            continue
        covered = [n for n in region(nodes, x0, x1)
                   if .3*n.size < n.base-y < .95*n.size]
        if not covered or x1-x0 > 100:
            continue
        largest = max(n.size for n in covered)
        baseline = min(n.base for n in covered if n.size >= largest*.9)
        # An overline can cover e^z as a whole. Keep its small raised z too.
        covered = [n for n in covered if abs(n.base-baseline) < largest*.9]
        text = r'\bar{'+math_sequence(covered)+'}'
        if text == r'\bar{lim}':
            text = r'\limsup '
        first = min(covered, key=lambda n: n.x0)
        new = replace(first, text=text, x0=min(n.x0 for n in covered),
                      x1=max(n.x1 for n in covered), y0=min(y, first.y0))
        nodes = [n for n in nodes if n not in covered]+[new]
        audit['overbars'] += 1
    for x0, x1, y in sorted(fractions, key=lambda r: r[1]-r[0]):
        column = region(nodes, x0, x1)
        near = [n for n in column if .15*n.size < y-n.base < .7*n.size]
        under = [n for n in column if .72*n.size < n.base-y < 1.7*n.size]
        if not near or not under:
            audit['unresolved_fraction_rules'] += 1
            continue
        size = max(n.size for n in near+under)
        topbase = max(n.base for n in near)
        bottombase = min(n.base for n in under)
        numerator = [n for n in column if topbase-size*1.05 <= n.base < y-.1]
        denominator = [n for n in column if y+.1 < n.base <= bottombase+size*.95]
        used = set(numerator+denominator)
        new = Node(r'\frac{'+math_sequence(numerator)+'}{'+math_sequence(denominator)+'}',
                   x0, min(n.y0 for n in used), x1, max(n.y1 for n in used),
                   y+size*.25, size, True, 'fraction')
        nodes = [n for n in nodes if n not in used]+[new]
        audit['fractions'] += 1
    return nodes


def extract_lines(page, audit):
    nodes = restore_geometry(raw_nodes(page, audit), horizontal_rules(page), audit)
    # Display operator limits occupy separate PDF text rows but belong to the operator.
    for op in [n for n in nodes if n.display_limits and
               n.text in (r'\int ', r'\oint ', r'\sum ', r'\prod ')]:
        limits = [n for n in nodes if n is not op and n.math and n.size < op.size*.86 and
                  op.x0-5 < n.cx < op.x1+6 and .65*op.size < abs(n.base-op.base) < 2.1*op.size]
        for side, marker in (('down', '_'), ('up', '^')):
            values = [n for n in limits if (n.base > op.base) == (side == 'down')]
            if values:
                op.text = op.text.rstrip()+marker+'{'+math_sequence(values)+'}'
        nodes = [n for n in nodes if n not in limits]
    # Attach small scripts before grouping complete body baselines.
    full = [n for n in nodes if n.size >= 8.3 or not n.math]
    small = [n for n in nodes if n not in full]
    # An exponent such as n+1 is one script row, even when its last glyph is
    # farther than eight points from the base. Nested tiny indices stay inside it.
    script_rows = []
    for n in sorted(small, key=lambda n: (n.base, n.x0)):
        group = next((g for g in script_rows if abs(g[0].base-n.base) < 1.25 and
                      n.x0 <= max(x.x1 for x in g)+4 and
                      n.x1 >= min(x.x0 for x in g)-4), None)
        if group is None:
            script_rows.append([n])
        else:
            group.append(n)
    attachments = {}
    for group in script_rows:
        n = min(group, key=lambda n: n.x0)
        targets = [m for m in full if m.math and m.x0 <= n.x0+1 and
                   -m.size*.65 <= n.x0-m.x1 < 8 and .8 < abs(n.base-m.base) < 8]
        centered = [m for m in full if m.text.strip() in (r'\limsup', 'lim', r'\lim') and
                    m.x0-5 <= n.x0 <= m.x1+5 and .8 < abs(n.base-m.base) < 12]
        targets = centered or targets
        if targets:
            parent = min(targets, key=lambda m: (abs(n.x0-m.x1), abs(n.base-m.base)))
            marker = '^' if n.base < parent.base else '_'
            attachments.setdefault(parent, {}).setdefault(marker, []).extend(group)
        else:
            full.extend(group)
    # Assign both prime and subscript against the original glyph width first.
    # Mutating that width after a prime used to orphan the subscript beneath it.
    for parent, sides in attachments.items():
        for marker in ('^', '_'):
            values = sides.get(marker, [])
            if values:
                text = math_sequence(values)
                parent.text += text if marker == '^' and set(text) == {"'"} else marker+'{'+text+'}'
                parent.x1 = max(parent.x1, *(n.x1 for n in values))
    groups = []
    for n in sorted(full, key=lambda n: (n.base, n.x0)):
        candidates = [(abs(g['base']-n.base), g) for g in groups if abs(g['base']-n.base) < 1.6]
        if candidates:
            min(candidates, key=lambda t: t[0])[1]['nodes'].append(n)
        else:
            groups.append({'base': n.base, 'nodes': [n]})
    result = []
    for group in sorted(groups, key=lambda g: g['base']):
        row = sorted(group['nodes'], key=lambda n: n.x0)
        pieces, math, previous = [], [], None
        def flush():
            if math:
                pieces.append('$'+math_sequence(math)+'$')
                math.clear()
        for n in row:
            gap = n.x0-previous.x1 if previous else 0
            if n.math:
                if not math and previous and not previous.math:
                    pieces.append(' ')
                math.append(n)
            else:
                flush()
                if previous and (previous.math or gap > n.size*.13):
                    pieces.append(' ')
                pieces.append(n.text)
            previous = n
        flush()
        text = re.sub(r' +', ' ', ''.join(pieces)).strip()
        if text:
            if all(n.math for n in row):
                text = '$$'+math_sequence(row)+'$$'
            formula_like = any(n.math for n in row) and not re.search('[А-Яа-яЁё]', text)
            result.append({'text': text, 'x0': min(n.x0 for n in row), 'base': group['base'],
                           'math_only': formula_like})
    return result


def extract_document(path):
    import fitz
    doc = fitz.open(path)
    blocks, audits = [], []
    section = 'Содержание'
    for pi, page in enumerate(doc):
        audit = {'page': pi+1, 'fractions': 0, 'overbars': 0, 'radicals': 0,
                 'unresolved_radicals': 0,
                 'unknown_glyphs': Counter(), 'unresolved_fraction_rules': 0}
        lines = extract_lines(page, audit)
        paragraphs, current = [], []
        current_section = section
        for line in lines:
            text = line['text']
            heading = re.match(r'^(?:Теорема|Определение|Следствие|Замечание|Доказательство|Пример)\b', text)
            sect = re.match(r'^(?:§\s*)?\d+(?:\.\d+)*\.\s+[А-Я]', text)
            paragraph = not line['math_only'] and line['x0'] > 51
            if current and (heading or sect or paragraph or sum(len(s) for s in current) > 1600):
                paragraphs.append((current_section, current))
                current = []
            if sect and pi > 1:
                section = text
            if not current:
                current_section = section
            # Remove a TeX line-break hyphen only between Cyrillic letters.
            if current and re.search(r'[а-яё]-$', current[-1]) and re.match('[а-яё]', text):
                current[-1] = current[-1][:-1]+text
            else:
                current.append(text)
        if current:
            paragraphs.append((current_section, current))
        for bi, (block_section, lines_in_block) in enumerate(paragraphs, 1):
            blocks.append({'block_id': f'p{pi+1:04d}_b{bi:03d}', 'page': pi+1,
                           'section': block_section, 'text': '\n'.join(lines_in_block)})
        audit['unknown_glyphs'] = dict(audit['unknown_glyphs'])
        audits.append(audit)
    return blocks, {'pages': len(doc), 'blocks': len(blocks), 'page_audit': audits,
        'method': 'Type3 cp1251 recovery + joint script rows + vector fractions/overbars/radicals',
        'limitations': 'Geometric heuristics, not exact TeX recovery; complex radicals/matrices may need review.'}
