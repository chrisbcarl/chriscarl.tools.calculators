#!/usr/bin/env python
# coding: utf-8
'''
Author:      Chris Carl
Date:        2025-01-30
Email:       chrisbcarl@outlook.com

Description:
    Calculators are straightforward in parlance but tricky in computation and solved by converting their format from infix to postfix or prefix
    The main algorithm is Edsger Dijkstra's Shunting Yard, which I shamelessly ripped for this one.
    https://en.wikipedia.org/wiki/Shunting_yard_algorithm
    A discrete calculator is different but the same, it just has different:
        - operators and precedents
        - parenthetical rules
        - value is a truth table
        - has no built-in eval
        - etc

Examples:
    python calculators/discrete.py "p" "( q )" "!p" "p & q" "p | q" "p -> q" "p iff q" "p -> ~q" "p & q | r"
    python calculators/discrete.py "a & not b" --formats csv --latex --output-filepath "ignoreme/out.csv"
    python calculators/discrete.py "p and q or r" "(p & q) <-> (r imp -s)" --formats html md json csv --latex --expand --output-filepath "ignoreme/out.txt"
    python calculators/discrete.py "(p & q) | (r and -s) iff t and not u"
    python calculators/discrete.py "(p & q) | (r and -s) iff (t and not u) implies v"

Warnings:
    piping can fail if the shell it's running in is not in utf-8 encoding or higher ex) python calculators/discrete.py > out.file  # on windows fails due to cp1252

Updated:
    2025-02-01 - chriscarl - discrete added --latex column output, csv, html katex, json, etc.
                             discrete hardened against bad eggs
                             discrete FIX: had a few false negatives (as of 01:04)
    2025-01-31 - chriscarl - discrete prettified to a BARE minimum
                             discrete single arrows
    2025-01-30 - chriscarl - discrete initial commit

TODO:
- tripple check the tt's
- if implies, add the inverse, converse, contrapositive
- add .tex latex, pass files in
- test dirty input like mismatched parens and make sure it catches, throw it bad stuff and make sure it can pick up that oh, this isnt an operator, or you threw two preps at once, or two ops in a row, etc.
- maybe simplify the logic somewhere, surely I can just evaluate based on the reverse polish notation straight away, i just didnt because i need to build up to the final,
    - all i have to do is compute (hidden) not values and just dont display them if not requested in the expression

Notes:
- 2025-01-30 18:00 - started
- 2025-01-30 19:50 - infix-postfix development ended
- 2025-01-30 20:44 - truth table filled with dummies, no calcs, paused
- 2025-01-30 23:30 - resumed after eating food
- 2025-01-31 01:16 - hard logical work done (not the display logics or appification), elapsed time: 01:16 - 23:30 + 20:44-19:50 (roughly 4 hrs of work and its done...)
- 2025-01-31 01:24 - markdown like 5 mins later 01:24
- 2025-01-31 01:30 - t0html like 5 mins later 01:30
'''
# stdlib
import os
import sys
import io
import re
import copy
import json
import csv
import pprint
import argparse
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

SCRIPT_DIRPATH = os.path.dirname(__file__)
SCRIPT_FILEPATH = __file__
SCRIPT_FILENAME = os.path.splitext(os.path.basename(SCRIPT_FILEPATH))[0]

OPERATORS = {
    'NOT': ['¬', '-', '!', '~', 'not', 'neg'],
    'CON': ['∧', '&', '*', 'and'],
    'DIS': ['∨', '|', '+', 'or'],
    'IMP': ['→', '⇒', '⇾', '->', '=>', 'implies', 'imp'],
    'IFF': ['↔', '⇔', '⇿', '<->', '<=>', 'iff'],
}
UNICODE_OPERATORS = {k: v[0] for k, v in OPERATORS.items()}
LATEX = {
    'NOT': '\\lnot',
    'CON': '\\land',
    'DIS': '\\lor',
    'IMP': '\\rightarrow',
    'IFF': '\\leftrightarrow',
}
UTF8 = {k: ord(v) for k, v in UNICODE_OPERATORS.items()}
PRECEDENCE = {
    0: 'NOT',
    1: 'CON',
    2: 'DIS',
    3: 'IMP',
    4: 'IFF'
}
HTML_FMT = '&#x{hex};'
OPERATOR_TO_PRECEDENCE = {v: k for k, v in PRECEDENCE.items()}
TRUTH_TABLES = {
    'NOT': {
        (True, ): False,
        (False, ): True,
    },
    'CON': {
        (True, True): True,
        (True, False): False,
        (False, True): False,
        (False, False): False,
    },
    'DIS': {
        (True, True): True,
        (True, False): True,
        (False, True): True,
        (False, False): False,
    },
    'IMP': {
        (True, True): True,
        (True, False): False,
        (False, True): True,
        (False, False): True,
    },
    'IFF': {
        (True, True): True,
        (True, False): False,
        (False, True): False,
        (False, False): True,
    },
}


def truth(p, op, /, q=None, default=(None, None)):  # syntax for positional passable, but the previous are definitley posiitonal
    if q is None:
        return TRUTH_TABLES.get(op, default)[(p,)]
    return TRUTH_TABLES.get(op, default)[(p, q)]


def sanitize_expression(expression):
    expression = re.sub(r'[\(\)]', '', expression)
    for difficulty in sorted(PRECEDENCE, reverse=True):  # simply because finding <-> needs to be replaced before ->
        op = PRECEDENCE[difficulty]
        tokens = OPERATORS[op]
        for token in tokens:
            expression = re.sub(re.escape(token), f' {op} ', expression)
    expression = re.sub(r'\s{1,}', ' ', expression)
    return expression


def prettify_expression(expression):
    # add back parens, use pretty symbols, etc.
    tokens = expression.split()
    if len(tokens) == 1:
        return tokens[0]

    groups = []
    left, right = -1, -1
    for t, token in enumerate(tokens):
        if token in OPERATORS:
            continue
        if left == -1:
            left = t
        elif right == -1:
            right = t + 1
            groups.append((left, right))
            left, right = -1, -1

    for left, right in reversed(groups):
        group = tokens[left:right]
        g = 0
        while g < len(group) - 1:
            if group[g] != 'NOT':  # to avoid (r ∨ ¬ s)
                group.insert(g + 1, ' ')
                g += 1
            g += 1
        tokens = tokens[:left] + [f'({"".join(group)})'] + tokens[right:]
    for t, token in enumerate(tokens):
        for op, lst in OPERATORS.items():
            token = re.sub(re.escape(op), lst[0], token)
        tokens[t] = token
    pretty = ' '.join(tokens)
    return re.sub('¬ ', '¬', pretty)  # a bit hacky, but works TODO: improve this case '¬ s', '(p ∨ q) ∧ (r ∨ ¬s)']


def shunting_yard(expression, verbose=False):
    '''stolen from wikipedia
    # FEATURE: shunting-yard-algorithm
    '''
    sanitized = sanitize_expression(expression)
    if verbose:
        print(expression, sanitized)
    output_queue = []
    operator_stack = []

    tokens = sanitized.split()
    for t, token in enumerate(tokens):
        if token not in OPERATORS:
            output_queue.append(token)
        else:
            op, prec = token, OPERATOR_TO_PRECEDENCE.get(token, -1)
            if operator_stack:
                stack_op = operator_stack[-1]
                while OPERATOR_TO_PRECEDENCE[stack_op] < prec:
                    if not operator_stack:
                        break
                    stack_op = operator_stack.pop()
                    output_queue.append(stack_op)
                    if not output_queue:
                        break
            operator_stack.append(op)

        if verbose:
            print('\t', t, token, output_queue, operator_stack)

    while operator_stack:
        op = operator_stack.pop()
        output_queue.append(op)

    if verbose:
        print(expression, '===(reverse polish notation)===', output_queue)
    return output_queue


T_TRUTH_TABLE = Dict[str, List[Optional[bool]]]


def create_truth_value_arrays(propositions):
    # type: (List[str]) -> Tuple[T_TRUTH_TABLE, int]
    power = len(propositions)
    total_truth_values = 2**power
    truth_table: T_TRUTH_TABLE = OrderedDict()  # with the final expression added on last.
    for p, prop in enumerate(propositions):
        truth_table[prop] = []
        ts_or_fs = 2**(power - p - 1)
        oscs = list(range(2**(p + 1)))
        for osc in oscs:
            if osc % 2 == 0:
                truth_table[prop].extend([True] * ts_or_fs)
            else:
                truth_table[prop].extend([False] * ts_or_fs)
    return truth_table, total_truth_values


def expression_into_logical_atomic_truth_columns(expression):
    '''take p AND NOT q -> [p, AND, NOT q] '''
    logical = []
    tokens = expression.split()
    while tokens:
        token = tokens.pop(0)
        if token == 'NOT':
            logical.append(f'{token} {tokens.pop(0)}')  # get the next as well
        else:
            logical.append(token)
    return logical


def _extract_propositions(expression):
    '''from p AND NOT q, return p, q
    requires it formatted in p OP q format'''
    propositions = []
    for token in expression.split():
        if token in OPERATORS:
            continue
        if token not in propositions:
            propositions.append(token)
    return propositions


def _evaluate_bool_op_list(bool_op_list, copyfirst=False):
    # type: (List[bool], bool) -> bool
    if copyfirst:
        bool_op_list = copy.deepcopy(bool_op_list)
    while bool_op_list:
        p = bool_op_list.pop(0)
        if not bool_op_list:
            return p
        op = bool_op_list.pop(0)
        q = bool_op_list.pop(0)
        tv = truth(p, op, q)

        bool_op_list.insert(0, tv)
    return False


def _has_bad_multi_op(original, expression, debug=False):
    tokens = expression.split()
    not_i = -1
    op_i = -1
    for t, token in enumerate(tokens):
        if token == 'NOT':
            if not_i != -1:
                if t > not_i + 1:
                    # situation where we have very loooong expressions 'p CON q DIS r CON NOT s IFF t CON NOT u'
                    not_i = t
                else:
                    # situation where we have NOT NOT p, NOT AND p
                    if debug:
                        print(original, repr(expression))
                    print(f'ERROR: unable to parse expression {original!r} because of multiple consecutive NOT operators!', file=sys.stderr)
                    sys.exit(1)
            not_i = t
        elif token in OPERATORS:
            if op_i != -1 and t == op_i + 1:
                # situation where we have AND OR p
                if debug:
                    print(original, repr(expression))
                print(f'ERROR: unable to parse expression {original!r} because of multiple non-NOT operators!', file=sys.stderr)
                sys.exit(1)
            op_i = t
    return


def evaluate(expression, rpn, verbose=True, debug=False):
    # type: (str, list, bool) -> T_TRUTH_TABLE
    '''cooking this one myself
    the "evaluation" is different because they're predicated on "building up" the truth table based on the tiniest first.
    (p & q) | (r and -s) ======== ['p', 'q', 'CON', 'r', 's', 'NOT', 'CON', 'DIS']
    i see p, q, then CON, so new is
        pNq, r, s, NOT, CON, DIS
    i see, pNq, r, s
        nothing
    i see r, s, NOT, so new is
        ...
    '''
    original_expression, original_rpn = expression, copy.deepcopy(rpn)
    propositions = []

    for token in rpn:
        if token in OPERATORS:
            continue
        if token not in propositions:
            propositions.append(token)

    truth_table, total_truth_values = create_truth_value_arrays(propositions)

    expressions = []
    runaway = 0
    while rpn:
        if len(rpn) == 1:
            expressions.append(rpn[0])
            break
        i = 0
        while i < len(rpn):
            token = rpn[i]
            if token in OPERATORS:
                if token == 'NOT':
                    # operator of one argument
                    operator, operand_right = rpn.pop(i), rpn.pop(i - 1)
                    # _has_bad_multi_op(original_expression, operator)
                    # _has_bad_multi_op(original_expression, operand_right)
                    expression = f'{operator} {operand_right}'
                    _has_bad_multi_op(original_expression, expression, debug=debug)

                    expressions.append(expression)
                    rpn.insert(i - 1, str(expression))
                    break
                else:
                    # operator of two arguments
                    # pop order matters, thats all
                    try:
                        operator, operand_right, operand_left = rpn.pop(i), rpn.pop(i - 1), rpn.pop(i - 2)
                        # _has_bad_multi_op(original_expression, operand_left)
                        # _has_bad_multi_op(original_expression, operator)
                        # _has_bad_multi_op(original_expression, operand_right)
                        expression = f'{operand_left} {operator} {operand_right}'
                        _has_bad_multi_op(original_expression, expression, debug=debug)
                        expressions.append(expression)
                        rpn.insert(i - 2, str(expression))
                        break
                    except IndexError:
                        print(f'ERROR: unable to parse expression {original_expression!r} due to malformed operators somewhere!', file=sys.stderr)
                        sys.exit(1)
            i += 1
        runaway += 1
        # if runaway > len(original_rpn) ** 2 - 1:
        #     print('wtf')
        if runaway > len(original_rpn) ** 2:
            print(f'ERROR: unable to parse expression {original_expression!r} due to {runaway} runaway iterations! Check malformat?', file=sys.stderr)
            sys.exit(1)
            # raise RecursionError()
    for expression in expressions:
        if expression in truth_table:
            continue  # something like ( q ) would already exist as q
        tokens = expression_into_logical_atomic_truth_columns(expression)

        truth_values = []  # type: List[Optional[bool]]
        for i in range(total_truth_values):
            expression_in_truth_values_and_ops = []
            for token in tokens:
                if token in OPERATORS:
                    expression_in_truth_values_and_ops.append(token)
                else:
                    subtokens = token.split()
                    if subtokens[0] == 'NOT':
                        p = subtokens[1]
                        value = not truth_table[p][i]
                    else:
                        p = subtokens[0]
                        value = truth_table[p][i]
                    expression_in_truth_values_and_ops.append(value)
            value = _evaluate_bool_op_list(expression_in_truth_values_and_ops, copyfirst=verbose)
            if verbose:
                print(original_expression, expression, tokens, expression_in_truth_values_and_ops, value)

            truth_values.append(value)
        truth_table[expression] = truth_values
    return truth_table


def pretty_expressions_to_latex(expressions, start='$', stop='$'):
    # type: (List[str], str, str) -> List[str]
    latexes = []
    for expression in expressions:
        for op, symbol in UNICODE_OPERATORS.items():
            tex = LATEX[op]
            expression = expression.replace(symbol, tex)
        latex = f'{(start + " ") if start else ""}{expression}{(" " + stop) if stop else ""}'
        latex = latex.replace('lnot', 'lnot ')  # otherwise you get \lnotp which is no good
        latexes.append(latex)
    return latexes


def to_markdown(tt, latex=False):
    # type: (Dict[str, List[bool]], bool) -> str
    headers = expressions = list(tt.keys())
    total_tvs = len(tt[expressions[0]])
    if latex:
        headers = pretty_expressions_to_latex(expressions, start='$', stop='$')

    lines = [
        f'| {" | ".join(headers)} |',
        f'| {" | ".join("-" * len(head) for head in headers)} |',
    ]
    for r in range(total_tvs):
        line = f'| {" | ".join(str(tt[exp][r])[0] + " " * (len(headers[e]) - 1) for e, exp in enumerate(expressions))} |'
        lines.append(line)
    return '\n'.join(lines)


def to_html(tt, expand=True, latex=False):
    # type: (Dict[str, List[bool]], bool, bool) -> str
    headers = expressions = list(tt.keys())
    total_tvs = len(tt[expressions[0]])
    if latex:
        headers = pretty_expressions_to_latex(expressions, start="$$", stop='$$')
    tokens = ['<table>', '    <thead>', '        <tr>']
    for e, _ in enumerate(expressions):
        tokens.append(f'            <th scope="col">{headers[e]}</th>')
    tokens += ['        </tr>', '    </thead>', '    <tbody>']
    for r in range(total_tvs):
        tokens.append('        <tr>')
        for exp in expressions:
            tokens.append(f'            <td>{str(tt[exp][r])[0]}</td>')
        tokens.append('        </tr>')
    tokens += ['    </tbody>', '</table>']
    if latex:
        # katex rather than mathjax, so we need to add a bunch of boilerplate so that it actually runs...
        # https://stackoverflow.com/a/65540803
        tokens = [
'<!DOCTYPE html>',
'<html>',
'    <head>',
'        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css" integrity="sha384-zh0CIslj+VczCZtlzBcjt5ppRcsAmDnRem7ESsYwWwg3m/OaJ2l4x7YBZl9Kxxib" crossorigin="anonymous">',
'        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js" integrity="sha384-Rma6DA2IPUwhNxmrB/7S3Tno0YY7sFu9WSYMCuulLhIqYSGZ2gKCJWIqhBWqMQfh" crossorigin="anonymous"></script>',
'        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js" integrity="sha384-hCXGrW6PitJEwbkoStFjeJxv+fSOOQKOPbJxSfM6G5sWZjAyWhXiTIIAmQqnlLlh" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>',
'    </head>',
'    <body>',
        ] + [
            f'        {tag}' for tag in tokens
        ] + [
'    </body>',
# https://stackoverflow.com/a/56038155
# https://katex.org/docs/autorender.html
'<script>',
'    document.addEventListener("DOMContentLoaded", function() {',
'        renderMathInElement(document.body, {',
'          // customised options',
'          // auto-render specific keys, e.g.:',
'          delimiters: [',
'              {left: "$$", right: "$$", display: true},',
'              {left: "$", right: "$", display: false},',
'              {left: "\\(", right: "\\)", display: false},',
'              {left: "\\[", right: "\\]", display: true}',
'          ],',
'          // rendering keys, e.g.:',
'          throwOnError : false',
'        });',
'    });',
'</script>',
'</html>',
        ]
    return ('\n' if expand else '').join([token if expand else token.strip() for token in tokens])


def to_json(tt, latex=False, expand=True):
    # type: (Dict[str, List[bool]], bool, bool) -> str
    headers = expressions = list(tt.keys())
    if latex:
        headers = pretty_expressions_to_latex(expressions, start='', stop='')
    return json.dumps({headers[k]: tt[key] for k, key in enumerate(tt)}, indent=4 if expand else None)


def to_csv(tt, latex=False):
    # type: (Dict[str, List[bool]], bool) -> str
    headers = expressions = list(tt.keys())
    total_tvs = len(tt[expressions[0]])
    if latex:
        headers = pretty_expressions_to_latex(expressions, start='', stop='')
    si = io.StringIO(newline='')
    writer = csv.writer(si)
    writer.writerow(headers)
    for r in range(total_tvs):
        row = [str(tt[exp][r])[0] for exp in expressions]
        writer.writerow(row)
    return '\n'.join([line for line in si.getvalue().splitlines() if line])


def to_latex(tt, expand=True):
    raise NotImplementedError('i havent tried actually making a proper .tex document by hand yet. check back later.')


class NiceFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


FORMATS = ['md', 'html', 'json', 'csv', 'latex']
DEFAULT_FORMATS = [FORMATS[0]]


def get_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(prog=SCRIPT_FILENAME, description=__doc__, formatter_class=NiceFormatter)
    parser.add_argument('expressions', type=str, nargs='+', help='math expressions, you can pass multiple with quotes')
    parser.add_argument('--verbose', '-v', action='store_true', help='show work on the way to answering the question')
    parser.add_argument('--debug', '-d', action='store_true', help='show LOTS of work on the way to answering the question')
    parser.add_argument('--formats', '-f', type=str, nargs='+', default=DEFAULT_FORMATS, choices=FORMATS, help='chose output formats, multiple supported')
    parser.add_argument('--expand', action='store_true', help='by default, output is minified')
    parser.add_argument('--latex', action='store_true', help='output the symbols in md, html, json AS LaTeX?')
    parser.add_argument('--output-filepath', type=str, help='provide an output file? same output in console goes to the file')
    return parser


def multi_print(msg, fps=None):
    fps = fps or [sys.stdout]
    for fp in fps:
        print(msg, file=fp)


def main(expressions, verbose=False, debug=False, formats=None, expand=True, latex=False, output_filepath=None):
    # type: (List[str], bool, bool, Optional[List[str]], bool, bool, Optional[str]) -> int
    formats = formats or DEFAULT_FORMATS
    fp = None
    if output_filepath:
        dirname = os.path.dirname(output_filepath)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        fp = open(output_filepath, 'w', encoding='utf-8')
    fps = [sys.stdout] if fp is None else [sys.stdout, fp]

    for expression in expressions:
        if verbose:
            multi_print(expression, fps=fps)
        rpn = shunting_yard(expression, verbose=verbose)
        if verbose:
            multi_print(f'\tinfix to postfix:   {rpn}', fps=fps)
        tt = evaluate(expression, rpn, verbose=verbose, debug=debug)
        if debug:
            pprint.pprint(tt, indent=2, width=999999)
        if verbose:
            multi_print(f'\traw decomposition:  {list(tt.keys())}', fps=fps)
        tt_pretty = OrderedDict([(prettify_expression(k), v) for k, v in tt.items()])
        for fmt in formats:
            if fmt == 'md':
                out = to_markdown(tt_pretty, latex=latex)
            elif fmt == 'html':
                out = to_html(tt_pretty, latex=latex, expand=expand)
            elif fmt == 'json':
                out = to_json(tt_pretty, latex=latex, expand=expand)
            elif fmt == 'csv':
                out = to_csv(tt_pretty, latex=latex)
            elif fmt == 'latex':
                out = to_latex(tt_pretty, expand=expand)
            multi_print(out, fps=fps)
    if fp:
        fp.close()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    sys.exit(main(args.expressions, verbose=args.verbose, debug=args.debug, formats=args.formats, expand=args.expand, latex=args.latex, output_filepath=args.output_filepath))


# confirmed bad eggs:
# python calculators/discrete.py "and p"
# python calculators/discrete.py "and and p"
# python calculators/discrete.py "and not p"
# python calculators/discrete.py "not not p"
# python calculators/discrete.py "not not p and"
# python calculators/discrete.py "p and"
# python calculators/discrete.py "p and not"
# python calculators/discrete.py "p and or"
# python calculators/discrete.py "p and or p"
