from __future__ import annotations

from itertools import count
from typing import Any

from backend_py.config import Config
from sympy import Symbol, cse, simplify
from sympy.utilities.lambdify import lambdify


class BasicBlock:
    """
    A run of statements without control flow.

    All statements can be reordered or changed to improve performance.
    """

    def __init__(self, *, arglist: list[str], statements: list[Any], config: Config):
        self._arglist = arglist
        self._exprs = statements
        self._config = config

        self._compile()

    def __len__(self):
        return len(self._exprs)

    def _compile(self):
        prefix = []
        body = self._exprs

        if self._config.common_subexpression_elimination:
            prefix, body = cse(body, symbols=(Symbol(f"_t{i}") for i in count()))

        temporaries = [r[0] for r in prefix]
        self._prefix = []
        for i in range(len(prefix)):
            expr = prefix[i][1]
            if self._config.common_subexpression_elimination:
                expr = simplify(expr)

            self._prefix.append(
                (
                    temporaries[i],
                    lambdify(
                        self._arglist + temporaries[:i],
                        expr,
                        modules=self._config.python_modules,
                        cse=False,
                    ),
                )
            )

        self._body = [
            lambdify(
                self._arglist + temporaries,
                (
                    simplify(expr)
                    if self._config.common_subexpression_elimination
                    else expr
                ),
                modules=self._config.python_modules,
                cse=False,
            )
            for expr in body
        ]

    def execute(self, *args, **kwargs):
        # Note: The list of statements is ordered and can get CSE or reordered within the block because we know it is straight calculation without control flow (a basic block)
        temporary_values = {}
        for name, expr in self._prefix:
            temporary_values[str(name)] = expr(*args, **kwargs, **temporary_values)

        for impl in self._body:
            yield impl(*args, **kwargs, **temporary_values)
