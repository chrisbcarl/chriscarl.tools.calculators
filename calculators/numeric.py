'''
Author:      Chris Carl
Date:        2025-01-30
Email:       chrisbcarl@outlook.com

Description:
    Calculators are straightforward in parlance but tricky in computation and solved by converting their format from infix to postfix or prefix
    The main algorithm is Edsger Dijkstra's Shunting Yard, which I shamelessly ripped for this one.
    https://en.wikipedia.org/wiki/Shunting_yard_algorithm

Examples:
    python calculators/numeric.py "-1"
    python calculators/numeric.py "-1" "1 + 1" "1 + (2 - 3) * (4 / 5) ** 6" "16 ^ 27 // (3 ** 2)"

Updated:
    2025-01-31 - chriscarl - numeric prettified to a BARE minimum
                             numeric FEATURE: numeric-pow-intdiv
    2025-01-30 - chriscarl - numeric initial commit

Notes:
- 2025-01-30 18:00 - started
- 2025-01-30 19:50 - main development ended. it just worked
'''
# stdlib
import os
import sys
import re
import copy
import argparse
from typing import List

SCRIPT_DIRPATH = os.path.dirname(__file__)
SCRIPT_FILEPATH = __file__
SCRIPT_FILENAME = os.path.splitext(os.path.basename(SCRIPT_FILEPATH))[0]

PRECENDENCE_TO_OPERATORS = {
    0: [
        '**', '//',  # FEATURE: numeric-pow-intdiv
        '*', '/'
    ],
    1: ['+', '-'],
    2: ['^'],
}
OPERATOR_TO_PRECEDENCE = {op: k for k, items in PRECENDENCE_TO_OPERATORS.items() for op in items}
OPERATORS = set(OPERATOR_TO_PRECEDENCE)

def sanitize_expression(expression):
    # type: (str) -> str
    expression = re.sub(r'([\(\)])', r' \g<1> ', expression)
    expression = re.sub(r'\s{1,}', ' ', expression)
    return expression


def shunting_yard(expression, verbose=False):
    # type: (str, bool) -> List[str]
    '''
    # FEATURE: shunting-yard-algorithm
    https://en.wikipedia.org/wiki/Shunting_yard_algorithm
    /* The functions referred to in this algorithm are simple single argument functions such as sine, inverse or factorial. */
    /* This implementation does not implement composite functions, functions with a variable number of arguments, or unary OPERATORS. */

    while there are tokens to be read:
        read a token
        if the token is:
        - a number:
            push it into the output_queue
        - a function:
            push it onto the operator_stack
        - an operator o1:
            while (
                there is an operator o2 at the top of the operator_stack which is not a left parenthesis,
                and (o2 has greater precedence than o1 or (o1 and o2 have the same precedence and o1 is left-associative))
            ):
                pop o2 from the operator_stack into the output_queue
            push o1 onto the operator_stack
        - a ",":
            while the operator at the top of the operator_stack is not a left parenthesis:
                pop the operator from the operator_stack into the output_queue
        - a left parenthesis (i.e. "("):
            push it onto the operator_stack
        - a right parenthesis (i.e. ")"):
            while the operator at the top of the operator_stack is not a left parenthesis:
                {assert the operator_stack is not empty}
                /* If the stack runs out without finding a left parenthesis, then there are mismatched parentheses. */
                pop the operator from the operator_stack into the output_queue
            {assert there is a left parenthesis at the top of the operator_stack}
            pop the left parenthesis from the operator_stack and discard it
            if there is a function token at the top of the operator_stack, then:
                pop the function from the operator_stack into the output_queue

    /* After the while loop, pop the remaining items from the operator_stack into the output_queue. */
    while there are tokens on the operator_stack:
        /* If the operator token on the top of the stack is a parenthesis, then there are mismatched parentheses. */
        {assert the operator on top of the stack is not a (left) parenthesis}
        pop the operator from the operator_stack onto the output_queue
    '''
    expression = sanitize_expression(expression)
    output_queue = []
    operator_stack = []

    # type_to_braces = {0: ['(', ')'], 1: ['[', ']'], 2: ['{', '}']}
    # brace_to_type = {op: k for k, items in type_to_braces.items() for op in items}
    # brace_orientation = {op: o for items in type_to_braces.values() for o, op in enumerate(items)}
    braces = ['(', ')']
    _operators = set(braces).union(OPERATOR_TO_PRECEDENCE)

    tokens = expression.split()
    for t, token in enumerate(tokens):
        bad_msg = 'unbalanced parenthesis at some point up to "{}"'.format(' '.join(tokens[0:t + 1]))
        if token not in _operators:
            output_queue.append(token)
        else:
            op, prec, is_brace = token, OPERATOR_TO_PRECEDENCE.get(token, -1), token in braces
            if is_brace:  # braces
                borient = braces.index(op)
                if borient == 0:  # left brace
                    operator_stack.append(op)
                else:  # right brace
                    if not operator_stack:
                        raise RuntimeError(bad_msg)
                    stack_op = operator_stack[-1]
                    while stack_op != '(':
                        stack_op = operator_stack.pop()
                        output_queue.append(stack_op)
                        if not operator_stack:
                            break
                        stack_op = operator_stack[-1]
                    # there HAS to be a left operator at this point, the one we were looking for, and discard it
                    stack_op = operator_stack.pop()
                    if stack_op != '(':
                        raise RuntimeError(bad_msg)
            else:  # other _operators
                if operator_stack:
                    stack_op = operator_stack[-1]
                    while stack_op != '(' and OPERATOR_TO_PRECEDENCE[stack_op] < prec:
                        stack_op = operator_stack.pop()
                        output_queue.append(stack_op)
                        if not output_queue:
                            break
                operator_stack.append(op)
        if verbose:
            print('\t', t, token, output_queue, operator_stack)
    bad_msg = 'unbalanced parenthesis at some point up to "{}"'.format(' '.join(tokens[0:t + 1]))
    while operator_stack:
        op = operator_stack.pop()
        if op in braces:
            raise RuntimeError(bad_msg)
        output_queue.append(op)

    return output_queue


def evaluate(rpn):
    # type: (List[str]) -> int | float
    '''cooking this one myself
    ( 3 + 5 ) + 4 + 9 ['3', '5', '+', '4', '9', '+', '+']
    i see 3, 5, then plus, so new is
        8, 4, 9, +, +
    i see 8, 4, nothing
    i see 4, 9, +, so new is
        8 13 +
    i see 8, 13, +, so new is
        21
    '''
    original = copy.deepcopy(rpn)
    while rpn:
        if len(rpn) == 1:
            return eval(rpn[0])
        elif len(rpn) == 2:
            raise ValueError('cant have reverse polish notation of only 2 tokens--that would imply that its 2 loose hanging numbers with no operator between them')
        i = 0
        while i < len(rpn):
            token = rpn[i]
            if token in OPERATORS:
                # pop order matters, thats all
                operator, operand_right, operand_left = rpn.pop(i), rpn.pop(i - 1), rpn.pop(i - 2)
                expression = f'{operand_left} {operator} {operand_right}'
                value = eval(expression)
                if value == int(value):
                    value = int(value)
                rpn.insert(i - 2, str(value))
                break
            i += 1
    return -1


class NiceFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def get_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(prog=SCRIPT_FILENAME, description=__doc__, formatter_class=NiceFormatter)
    parser.add_argument('expressions', type=str, nargs='+', help='math expressions, you can pass multiple with quotes')
    parser.add_argument('--verbose', '-v', action='store_true', help='show work on the way to answering the question')
    parser.add_argument('--compare', '-c', action='store_true', help='compare with eval(1 + 1)')
    return parser


def main(expressions, verbose=False, compare=False):
    # type: (List[str], bool, bool) -> int
    for expression in expressions:
        san = sanitize_expression(expression)
        if verbose:
            print(expression)
            print(f'\tsanitized:          {san}')
        # print(expression, reverse_polish_notation)
        rpn = shunting_yard(san)
        if verbose:
            print(f'\tinfix to postfix:   {rpn}')
        val = evaluate(rpn)
        if compare:
            try:
                eeval = eval(san)
                if verbose:
                    print(f'\tshunting_yard eval: {val}')
                    print(f'\tpython eval:        {eeval}')
                assert eeval == val, f'{expression} != {val}'
            except TypeError as te:
                print('couldnt evaluate using eval, got', te)
        if verbose:
            print(f'\tvalue:              {val}')
        else:
            print(val)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    sys.exit(main(args.expressions, verbose=args.verbose, compare=args.compare))
