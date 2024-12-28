import argparse
import logging
from collections import namedtuple
from dataclasses import dataclass
from itertools import count
from typing import Any, Iterable, List, Optional, Tuple

from formak.ast_tools import (
    BaseAst,
    ClassDef,
    CompileState,
    ForwardClassDeclaration,
    HeaderFile,
    MemberDeclaration,
    Namespace,
    Return,
    SourceFile,
)
from formak.exceptions import ModelConstructionError
from sympy import Symbol, ccode, cse, diff, simplify

from formak import ast_fragments as fragments
from formak import common


@dataclass
class Config:
    """
    Options for generating C++.

    common_subexpression_elimination:
        Remove common shared computation
    extra_validation:
        Catch errors earlier in exchange for increased compute time
    """

    common_subexpression_elimination: bool = True
    extra_validation: bool = False
    max_dt_sec: float = 0.1
    innovation_filtering: float = 5.0

    def ccode(self):
        if self.max_dt_sec < 1e-9:
            raise ValueError(
                "Please specify Config(max_dt_sec=...) >= 1e-9. Currently {self.max_dt_sec}"
            )

        return Namespace(
            name="cpp",
            body=[
                ClassDef(
                    "struct",
                    "Config",
                    bases=[],
                    body=[
                        MemberDeclaration(
                            "static constexpr bool",
                            "common_subexpression_elimination",
                            (
                                "true"
                                if self.common_subexpression_elimination
                                else "false"
                            ),
                        ),
                        MemberDeclaration(
                            "static constexpr bool",
                            "extra_validation",
                            "true" if self.extra_validation else "false",
                        ),
                        MemberDeclaration(
                            "static constexpr double", "max_dt_sec", self.max_dt_sec
                        ),
                        MemberDeclaration(
                            "static constexpr double",
                            "innovation_filtering",
                            (
                                self.innovation_filtering
                                if self.innovation_filtering
                                else 0.0
                            ),
                        ),
                    ],
                )
            ],
        )


