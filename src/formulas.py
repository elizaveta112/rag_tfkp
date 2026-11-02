"""Deterministic search views. Never use these views as quoted evidence."""
import re
import unicodedata

MATH = re.compile(
    r"(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$|\\\[[\s\S]*?\\\]|"
    r"\\\([\s\S]*?\\\)|(?<![\\$])\$(?!\$)(?:\\.|[^$\\])*?(?<!\\)\$(?!\$)|"
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}[\s\S]*?\\end\{\1\}")
UNICODE = {
    '∂': r'\partial ', '∮': r'\oint ', '∫': r'\int ', '∑': r'\sum ',
    '∞': r'\infty ', '≤': r'\le ', '≥': r'\ge ', '≠': r'\ne ',
    '≡': r'\equiv ', '→': r'\to ', '⇒': r'\Rightarrow ', '⇔': r'\Leftrightarrow ',
    '∈': r'\in ', '∉': r'\notin ', '⊂': r'\subset ', '⊆': r'\subseteq ',
    'ℂ': r'\mathbb{C}', 'ℝ': r'\mathbb{R}', 'π': r'\pi ', 'γ': r'\gamma ',
    'φ': r'\varphi ', 'ϕ': r'\phi ', 'ζ': r'\zeta ', 'Δ': r'\Delta ',
    '−': '-', '–': '-', '′': "'", '″': "''", '‴': "'''", '⋅': r'\cdot ',
    '²': '^{2}', '³': '^{3}', '⁴': '^{4}', '¹': '^{1}', '⁰': '^{0}',
}
UNICODE.update({
    'α': r'\alpha ', 'β': r'\beta ', 'δ': r'\delta ',
    'ε': r'\varepsilon ', 'ϵ': r'\epsilon ', 'η': r'\eta ',
    'θ': r'\theta ', 'ϑ': r'\vartheta ', 'ι': r'\iota ',
    'κ': r'\kappa ', 'λ': r'\lambda ', 'μ': r'\mu ', 'ν': r'\nu ',
    'ξ': r'\xi ', 'ο': 'o', 'ρ': r'\rho ', 'σ': r'\sigma ', 'ς': r'\sigma ',
    'τ': r'\tau ', 'υ': r'\upsilon ', 'χ': r'\chi ', 'ψ': r'\psi ', 'ω': r'\omega ',
    'Γ': r'\Gamma ', 'Θ': r'\Theta ', 'Λ': r'\Lambda ', 'Ξ': r'\Xi ',
    'Π': r'\Pi ', 'Σ': r'\Sigma ', 'Φ': r'\Phi ', 'Ψ': r'\Psi ', 'Ω': r'\Omega ',
})
OPS = {'+': 'plus', '-': 'minus', '=': 'equals', '!': 'factorial',
       '^': 'power', '_': 'subscript', '/': 'divide', '|': 'abs',
       "'": 'prime', '<': 'lt', '>': 'gt', '{': 'open', '}': 'close',
       '(': 'lparen', ')': 'rparen'}
STOP = set('и в во на по из для что это как ли при или с со к ко а но не'.split())


def _brace_arguments(text):
    """Normalize only TeX's explicit argument syntax, no symbolic algebra."""
    def argument(pos):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return None
        if text[pos] == '{':
            depth, end = 1, pos+1
            while end < len(text) and depth:
                if text[end] == '\\':
                    end += 2
                    continue
                depth += (text[end] == '{')-(text[end] == '}')
                end += 1
            return (text[pos:end], end) if depth == 0 else None
        if text[pos] == '\\':
            m = re.match(r'\\(?:[A-Za-z]+|.)', text[pos:])
            if not m:
                return None
            return '{'+m.group()+'}', pos+len(m.group())
        if text[pos] in '}$^_':
            return None
        return '{'+text[pos]+'}', pos+1

    result, pos = [], 0
    while pos < len(text):
        match = re.match(r'\\(frac|bar|mathbb)(?![A-Za-z])', text[pos:])
        if match:
            end = pos+len(match.group())
            args = []
            for _ in range(2 if match[1] == 'frac' else 1):
                value = argument(end)
                if value is None:
                    break
                token, end = value
                args.append(token)
            expected = 2 if match[1] == 'frac' else 1
            if len(args) == expected:
                result.append(match.group()+''.join(args))
                pos = end
                continue
        result.append(text[pos])
        pos += 1
    return ''.join(result)


def canonical(text):
    # Combining overlines are semantic: do this BEFORE Unicode composition.
    text = re.sub(r'([A-Za-z\u0370-\u03ff])[̄̅]', lambda m: r'\bar{' + m[1] + '}', text)
    text = ''.join(UNICODE.get(c, c) for c in text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\\(?:dfrac|tfrac)(?![A-Za-z])', r'\\frac', text)
    text = re.sub(r'\\overline\b', r'\\bar', text)
    text = re.sub(r'\\(neq|leq|geq)(?![A-Za-z])',
                  lambda m: '\\'+{'neq': 'ne', 'leq': 'le', 'geq': 'ge'}[m[1]], text)
    text = re.sub(r'\\(?:left|right|displaystyle|textstyle)\b', '', text)
    text = re.sub(r'\\(?:,|;|!|quad\b|qquad\b)', ' ', text)
    # Canonical braces for a one-letter overline; keep all operators.
    text = re.sub(r'\\bar\s+([A-Za-z])', lambda m: r'\bar{' + m[1] + '}', text)
    return _brace_arguments(text)


def math_tokens(text):
    text = canonical(text)
    lex = re.findall(r'\\[A-Za-z]+|[A-Za-z\u0370-\u03ff]|\d+|_|[^\w\s$]', text)
    tokens = ['m_' + (('cmd_' + t[1:]) if t.startswith('\\')
                     else OPS.get(t, t)) for t in lex]
    # Preserve ordering/sign/conjugation with local structural features.
    tokens += ['seq_' + '|'.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    tokens += ['bar_' + m for m in re.findall(r'\\bar\s*\{([A-Za-z])\}', text)]
    return tokens


def stem(word):
    word = word.lower().replace('ё', 'е')
    if len(word) > 5 and re.fullmatch('[а-я]+', word):
        for suffix in ('ениями', 'остью', 'ение', 'ания', 'енных', 'ыми', 'ами',
                       'ого', 'ему', 'ому', 'иях', 'ях', 'ые', 'ая', 'ой', 'ий',
                       'ый', 'ов', 'ам', 'ом', 'ах', 'ии', 'ия', 'ие', 'ую'):
            if word.endswith(suffix) and len(word)-len(suffix) >= 4:
                return word[:-len(suffix)]
    return word


def word_tokens(text):
    return [stem(w) for w in re.findall(r'[A-Za-zА-Яа-яЁё]+|\d+', text)
            if w.lower() not in STOP]


def char_tokens(text):
    # Prose morphology; formula structure has its OWN case-sensitive channel.
    result = []
    for w in word_tokens(text):
        if len(w) >= 4:
            w = '^' + w + '$'
            result.extend(w[i:i+3] for i in range(len(w)-2))
    return result


def formula_view(text):
    spans = [m.group() for m in MATH.finditer(text)]
    # Corpus may already be plain Unicode math without delimiters.
    lines = [line for line in text.splitlines()
             if re.search(r'\\(?:frac|partial|int|oint|sum|bar)|[=∂∫∮≤≥≠≡]', line)]
    return ' '.join(dict.fromkeys(spans + lines))


def safe_units(text):
    """Whitespace/sentence boundaries outside complete formula spans."""
    result, last = [], 0
    for match in MATH.finditer(text):
        result.extend(re.split(r'(?<=\s)', text[last:match.start()]))
        result.append(match.group())
        last = match.end()
    result.extend(re.split(r'(?<=\s)', text[last:]))
    return [s for s in result if s]


def chunk_text(text, max_chars=1400):
    current = ''
    for unit in safe_units(text):
        if current and len(current)+len(unit) > max_chars:
            yield current
            current = ''
        current += unit
    if current:
        yield current
