import sys
import unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_corpus import Node, decode_text, restore_geometry, math_sequence, extract_document


def audit():
    return {'unknown_glyphs': Counter(), 'overbars': 0, 'fractions': 0,
            'unresolved_fraction_rules': 0}


class PDFCorpusTests(unittest.TestCase):
    def test_cyrillic_encoding(self):
        self.assertEqual(decode_text('Òåîðåìà'), 'Теорема')
        self.assertEqual(decode_text('\x9f'), '§')

    def test_bar_is_recovered_from_rule_geometry(self):
        z = Node('z', 10, 10, 15, 20, 18, 10, True)
        a = audit()
        result = restore_geometry([z], [(10, 15, 12.3)], a)
        self.assertEqual(result[0].text, r'\bar{z}')
        self.assertEqual(a['overbars'], 1)

    def test_fraction_and_factorial(self):
        n = Node('n', 10, 10, 15, 20, 18, 10, True)
        bang = Node('!', 15, 10, 18, 20, 18, 10, True)
        d = Node('2', 12, 24, 17, 34, 31.57, 10, True)
        a = audit()
        result = restore_geometry([n, bang, d], [(10, 18, 22.25)], a)
        self.assertEqual(result[0].text, r'\frac{n!}{2}')
        self.assertEqual(a['fractions'], 1)

    def test_index_is_distinct_from_plain_letter(self):
        z = Node('z', 10, 10, 15, 20, 18, 10, True)
        zero = Node('0', 15, 15, 18, 22, 19.5, 7, True)
        self.assertEqual(math_sequence([z, zero]), 'z_{0}')

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'source/textbook.pdf').exists(),
                         'PDF is downloaded during prepare_data.py, not stored in the repository')
    def test_real_book_critical_formulas(self):
        from common import ROOT
        blocks, meta = extract_document(ROOT/'source/textbook.pdf')
        self.assertEqual(meta['pages'], 147)
        page12 = '\n'.join(b['text'] for b in blocks if b['page'] == 12)
        self.assertIn(r'\frac{\partial f}{\partial \bar{z}}=0', page12)
        self.assertIn(r'\frac{\partial f}{\partial x}+i\frac{\partial f}{\partial y}', page12)
        self.assertIn(r'\frac{\partial f}{\partial x}-i\frac{\partial f}{\partial y}', page12)
        page34 = '\n'.join(b['text'] for b in blocks if b['page'] == 34)
        self.assertIn(r'\frac{n!}{2\pi i}\int', page34)
        self.assertIn('(\\zeta -z)^{n+1}', page34)
        page36 = '\n'.join(b['text'] for b in blocks if b['page'] == 36)
        self.assertIn(r'\frac{f^{(n)}(z_{0})}{n!}', page36)
        page11 = '\n'.join(b['text'] for b in blocks if b['page'] == 11)
        self.assertIn("u'_{x}", page11)
        self.assertIn("v'_{y}", page11)
        page17 = '\n'.join(b['text'] for b in blocks if b['page'] == 17)
        self.assertIn(r'\limsup _{n\to \infty }\sqrt[n]{|a_{n}|}', page17)
        page20 = '\n'.join(b['text'] for b in blocks if b['page'] == 20)
        self.assertIn(r'\bar{e^{z}}=e^{\bar{z}}', page20)
        self.assertIn(r'e^{z_{1}+z_{2}}=e^{z_{1}}e^{z_{2}}', page20)
        page37 = '\n'.join(b['text'] for b in blocks if b['page'] == 37)
        self.assertIn(r'\frac{f(\zeta )}{\zeta ^{n+1}}', page37)


if __name__ == '__main__':
    unittest.main()
