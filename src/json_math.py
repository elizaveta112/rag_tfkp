"""Repair JSON control escapes that a model used for known TeX commands.

No mathematical inference, answer selection, source lookup or ID-based rules.
"""
import re

COMMANDS = set('frac dfrac tfrac bar overline mathbb bb text textbf textit textrm '
               'times to theta tau top beta begin big bigl bigr boldsymbol '
               'ne neq nu nabla not notin right rho rangle rbrace rm '
               'partial gamma varphi phi alpha delta epsilon lambda pi sum int oint'.split())
PREFIX = {'\b': 'b', '\f': 'f', '\t': 't', '\n': 'n', '\r': 'r'}


def restore_math_escapes(text):
    def control(match):
        command = PREFIX[match[1]]+match[2]
        return '\\'+command if command in COMMANDS else match.group()
    text = re.sub(r'([\x08\x0c\t\n\r])([A-Za-z]+)', control, text)
    # Overescaped commands also occur, e.g. slash + backspace + 'b'.
    def command(match):
        if match[1] not in COMMANDS:
            return match.group()
        return '\\'+('mathbb' if match[1] == 'bb' else match[1])
    return re.sub(r'\\+([A-Za-z]+)', command, text)
