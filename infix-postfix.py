# started 2025-01-30 18:00
# Edsger Dijkstra, shunting yard algorithm - https://en.wikipedia.org/wiki/Shunting_yard_algorithm
# ended 19:50
# TODO: intdiv vs div, might be worth truly "tokenizing" first, ** vs ^, ** has to be found first, etc. multichar stuff etc.
import re
import copy


wiki_pseudocode = '''
/* The functions referred to in this algorithm are simple single argument functions such as sine, inverse or factorial. */
/* This implementation does not implement composite functions, functions with a variable number of arguments, or unary operators. */

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


def sanitize_expression(expression):
    expression = re.sub(r'(\d+)\s*\(', r'\g<1> * (', expression)
    expression = re.sub(r'\)\s*(\d+)', r') * \g<1>', expression)
    return expression


def shunting_yard(expression, verbose=False):
    '''stolen from wikipedia'''
    expression = sanitize_expression(expression)
    output_queue = []
    operator_stack = []

    # type_to_braces = {0: ['(', ')'], 1: ['[', ']'], 2: ['{', '}']}
    # brace_to_type = {op: k for k, items in type_to_braces.items() for op in items}
    # brace_orientation = {op: o for items in type_to_braces.values() for o, op in enumerate(items)}
    braces = ['(', ')']
    precedence_to_operators = {
        0: ['*', '/'],
        1: ['+', '-'],
        2: ['^'],
    }
    operator_to_precedence = {op: k for k, items in precedence_to_operators.items() for op in items}
    operators = set(braces).union(operator_to_precedence)

    tokens = expression.split()
    for t, token in enumerate(tokens):
        bad_msg = 'unbalanced parenthesis at some point up to "{}"'.format(' '.join(tokens[0:t + 1]))
        if token not in operators:
            output_queue.append(token)
        else:
            op, prec, is_brace = token, operator_to_precedence.get(token, -1), token in braces
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
            else:  # other operators
                if operator_stack:
                    stack_op = operator_stack[-1]
                    while stack_op != '(' and operator_to_precedence[stack_op] < prec:
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
    # type: (list) -> int | float
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
    precedence_to_operators = {
        0: ['*', '/'],
        1: ['+', '-'],
        2: ['^'],
    }
    operator_to_precedence = {op: k for k, items in precedence_to_operators.items() for op in items}
    operators = set(operator_to_precedence)
    while rpn:
        if len(rpn) == 1:
            return eval(rpn[0])
        elif len(rpn) == 2:
            raise ValueError('cant have reverse polish notation of only 2 tokens--that would imply that its 2 loose hanging numbers with no operator between them')
        i = 0
        while i < len(rpn):
            token = rpn[i]
            if token in operators:
                # pop order matters, thats all
                operator, operand_right, operand_left = rpn.pop(i), rpn.pop(i - 1), rpn.pop(i - 2)
                expression = f'{operand_left} {operator} {operand_right}'
                value = eval(expression)
                rpn.insert(i - 2, str(value))
                break
            i += 1
    return -1


expression = 'p -> q'  # infix
# postfix = 'p q ->'  # postfix (reverse polish notation)

expressions = ['3', '( 3 )', '3 + 5', '( 3 + 5 )', '( 3 + 5 ) + 4 + 9', '( 25 / 5 ) + 9 * ( 4 + 6 )']
for expression in expressions:
    # san = sanitize_expression(expression)
    san = expression
    eeval = eval(san)
    reverse_polish_notation = shunting_yard(expression)
    print(expression, reverse_polish_notation)
    val = evaluate(reverse_polish_notation)
    print(expression, '=', val)
    assert eeval == val, f'{expression} != {val}'
    print('\n\n')
