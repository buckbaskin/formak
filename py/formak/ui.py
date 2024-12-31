try:
    from problemdefinition.model import Model
except ImportError:
    print("womp womp what do")
    import sys

    prev_line = ""
    for line in sys.path:
        print(line[-80:])
        prev_line = line

    print("")
    print("end")
    raise

from formak.ui_state_machine import DesignManager, NisScore, StateId
from sympy import Matrix, Symbol, simplify, symbols
