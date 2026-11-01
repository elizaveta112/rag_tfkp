import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from formulas import word_tokens, char_tokens, math_tokens, formula_view, canonical


def is_statement(text):
    return bool(re.match(r'^(?:Теорема|Следствие|Определение|Экспонента|Тригонометрические функции)\b', text))


def concept_queries(text):
    """Generic notation-to-topic lookup; independent of question IDs and gold."""
    text = canonical(text).lower()
    queries = []
    if re.search(r'e\s*\^\s*\{?z|экспонент', text):
        queries.append(('экспонента', r'^Экспонента\b'))
    if re.search(r'\\(?:int|oint)|∫|интеграл', text):
        queries.append(('голоморфная бесконечно дифференцируемой производные', r'голоморфн'))
    if re.search(r'нул(?:и|ей|ями|ях)\b|совпада|[fg]\s*\([^)]*\)\s*=\s*0', text):
        queries.append(('единственности предельную точку', r'предельн'))
    if re.search(r'\\sin|синус', text):
        queries.append(('тригонометрические функции', r'^Тригонометрические функции'))
    if re.search(r'тейлор|радиус сходимости', text) or ('голоморф' in text and 'разложени' in text):
        queries.append(('коэффициенты сходящегося степенного ряда', r'голоморфн'))
    if 'гармонич' in text and 'голоморф' in text:
        queries.append(('голоморфная бесконечно дифференцируемой', r'голоморфн'))
    return queries[:4]


def make_bm25(documents):
    postings = defaultdict(list)
    lengths = []
    for i, tokens in enumerate(documents):
        counts = Counter(tokens)
        lengths.append(len(tokens))
        for term, count in counts.items():
            postings[term].append([i, count])
    return {'postings': dict(postings), 'lengths': lengths,
            'average': sum(lengths)/max(1, len(lengths))}


def bm25(index, query, top_k=24):
    n, average = len(index['lengths']), index['average'] or 1
    scores = defaultdict(float)
    for term in set(query):
        entries = index['postings'].get(term, [])
        idf = math.log(1+(n-len(entries)+0.5)/(len(entries)+0.5))
        for i, tf in entries:
            length = index['lengths'][i]
            scores[i] += idf * tf * 2.2 / (tf+1.2*(0.25+0.75*length/average))
    return sorted(scores, key=lambda i: (-scores[i], i))[:top_k]


def build_views(chunks):
    texts = [str(c['section'])+'\n'+c['text'] for c in chunks]
    return {
        'word': make_bm25([word_tokens(s) for s in texts]),
        'char': make_bm25([char_tokens(s) for s in texts]),
        'math': make_bm25([math_tokens(formula_view(s)) for s in texts]),
        'statement': make_bm25([word_tokens(s) if is_statement(c['text']) else []
                                 for s, c in zip(texts, chunks)]),
    }


class Retriever:
    def __init__(self, path):
        with gzip.open(Path(path)/'index.json.gz', 'rt', encoding='utf-8') as f:
            self.data = json.load(f)
        self.chunks = self.data['chunks']
        self.blocks = self.data['blocks']
        self.manifest = json.loads((Path(path)/'manifest.json').read_text(encoding='utf-8'))

    def search(self, q, mode, settings):
        if mode == 'no_context':
            return [], []
        combined = q['question']+'\n'+'\n'.join(o['text'] for o in q['options'])
        if mode == 'single_rag':
            ranks = [(1., bm25(self.data['views']['word'], word_tokens(combined), 40))]
        else:
            # Fixed deterministic decomposition, no generated enrichment.
            queries = [(combined, 1.), (q['question'], 1.)]
            queries += [(q['question']+'\n'+o['text'], 0.5/len(q['options']))
                        for o in q['options']]
            # Fixed mathematical terminology, not generated facts or answer rules.
            # Links symbolic requests to theorem names used in the book.
            text = canonical(combined).lower()
            terminology = []
            if re.search(r'e\s*\^\s*\{?z|экспонент', text):
                terminology.append('экспонента показательная функция производная')
            if re.search(r'\\(?:int|oint)|∫|интеграл', text):
                terminology.append('интегральная формула Коши производные факториал')
            if 'гармонич' in text:
                terminology.append('условия Коши Римана сопряженные гармонические функции')
                if 'голоморф' in text:
                    terminology.append('голоморфная бесконечно дифференцируемая производные всех порядков')
            if re.search(r'нул[ьяе]|совпада', text):
                terminology.append('теорема единственности предельная точка целые функции синус')
            queries += [(t, .55) for t in terminology]
            ranks = []
            for query, weight in queries:
                ranks.append((weight, bm25(self.data['views']['word'], word_tokens(query))))
                ranks.append((0.35*weight, bm25(self.data['views']['char'], char_tokens(query))))
                if mode != 'no_formula':
                    math = formula_view(query)
                    if math:
                        ranks.append((0.8*weight, bm25(self.data['views']['math'], math_tokens(math))))
        scores = defaultdict(float)
        for weight, ranking in ranks:
            for rank, i in enumerate(ranking, 1):
                scores[i] += weight/(60+rank)
        anchors = []
        if mode != 'single_rag':
            if 'statement' not in self.data['views']:
                raise ValueError('Rebuild the index: statement coverage view is missing')
            for query, required in concept_queries(combined):
                # Reserve one formulation for each independent concept rather
                # than letting multiple similar proofs consume all eight blocks.
                for i in bm25(self.data['views']['statement'], word_tokens(query), 40):
                    if not re.search(required, self.chunks[i]['text'], re.IGNORECASE):
                        continue
                    if all(self.chunks[j]['block_id'] != self.chunks[i]['block_id'] for j in anchors):
                        anchors.append(i)
                        break
            for i in scores:
                if self.chunks[i]['text'].startswith('Доказательство'):
                    scores[i] *= .7
                elif is_statement(self.chunks[i]['text']):
                    scores[i] *= 1.15
        ordered = sorted(scores, key=lambda i: (-scores[i], i))
        chosen = list(dict.fromkeys(anchors+ordered))[:settings['seed_chunks']]
        # At most one nearby chunk per seed; preserves conditions across blocks.
        if mode not in ('single_rag', 'no_neighbors'):
            for i in chosen[:settings['neighbor_seeds']]:
                for j in (i-1, i+1):
                    if 0 <= j < len(self.chunks) and j not in chosen:
                        a, b = self.chunks[i], self.chunks[j]
                        if a['section'] == b['section'] and abs(a['page']-b['page']) <= 1:
                            scores[j] = max(scores[j], scores[i]*0.88)
            chosen = list(dict.fromkeys(anchors+sorted(scores, key=lambda i: (-scores[i], i))))[:settings['candidate_chunks']]
        return [dict(self.chunks[i], score=scores[i]) for i in chosen], ordered[:40]
