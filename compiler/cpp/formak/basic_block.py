from itertools import count
from typing import Any, List, Tuple

from formak.ast_tools import (
    MemberDeclaration,
)
from formak.compiler.comfig import Config
from sympy import Symbol, ccode, cse, simplify


class BasicBlock:
    """
    A run of statements without control flow.

    All statements can be reordered or changed to improve performance.
    """

    def __init__(
        self, *, statements: List[Tuple[str, Any]], indent: int, config: Config
    ):
        # should be Tuple[str, sympy expression]
        statements = list(statements)
        self._targets = [k for k, _ in statements]
        self._exprs = [v for _, v in statements]
        self._indent = indent
        self._config = config

    def __len__(self):
        return len(self._exprs)

    def compile(self):
        prefix = []
        body = self._exprs

        if self._config.common_subexpression_elimination:
            prefix, body = cse(body, symbols=(Symbol(f"_t{i}") for i in count()))

        # Note: The list of statements is ordered and can get CSE or reordered
        # within the block because we know it is straight calculation without
        # control flow (a basic block)
        for target, expr in prefix:
            assert isinstance(target, Symbol)
            if self._config.common_subexpression_elimination:
                expr = simplify(expr)
            cc_expr = ccode(expr)
            yield MemberDeclaration("double", target, cc_expr)

        for target, expr in zip(self._targets, body):
            if self._config.common_subexpression_elimination:
                expr = simplify(expr)
            cc_expr = ccode(expr)
            yield MemberDeclaration("", target, cc_expr)
