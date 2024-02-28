from __future__ import annotations
from collections import namedtuple
import sympy
from sympy import Symbol, Expr, symbols


class ActionBase:
    def cost(self) -> float:
        raise NotImplementedError()


class StateBase:
    def __eq__(self, other: StateBase) -> bool:
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()


FrontTuple = namedtuple("FrontTuple", ["state", "path", "reach", "heuristic"])


class Frontier:
    def __init__(self, initial_state):
        self.q = [FrontTuple(initial_state, [], 0.0, 0.0)]

    def append(
        self,
        state: StateBase,
        path: List[ActionBase],
        reach: float,
        heuristic: float,
        *,
        debug=False,
    ):
        self.q.append(FrontTuple(state, path, reach, heuristic))

        self.q.sort(key=lambda t: t.reach + t.heuristic)
        if debug:
            print(
                [
                    (t.state._name, t.reach, t.heuristic, t.reach + t.heuristic)
                    for t in self.q
                ]
            )

    def pop(self) -> StateBase:
        if len(self.q) == 0:
            raise ValueError("Empty Frontier")

        result = self.q[0]
        self.q = self.q[1:]

        return result


def astar(
    initial_state: StateBase,
    goal_state: StateBase,
    transition,
    heuristic,
    edges,
    *,
    max_iter=1000,
):
    frontier = Frontier(initial_state)
    visited = set()

    for i in range(max_iter):
        try:
            state, path, reach_cost, _ = frontier.pop()
        except ValueError:
            print("iter", i)
            raise

        # Note: put goal_state first so it can override equality (if desired)
        if goal_state == state:
            return path

        if state in visited:
            continue

        visited.add(state)

        for action in edges(state):
            next_state = transition(state, action)

            frontier.append(
                next_state,
                path + [action],
                reach_cost + action.cost,
                heuristic(next_state),
            )

    raise ValueError("Failed to terminate in max iterations")


class InstructionBase:
    pass


class Nop(InstructionBase):
    def __repr__(self):
        return "Nop()"

    def __eq__(self, other):
        if not isinstance(other, Nop):
            return False

        return True

    def __hash__(self):
        return hash("Nop")


class BinOpBase(InstructionBase):
    def __init__(self, l, r):
        self.l = l
        self.r = r


class Add(BinOpBase):
    def __repr__(self):
        return f"Add({self.l}, {self.r})"

    def __eq__(self, other):
        if not isinstance(other, Add):
            return False

        return (self.l, self.r) == (other.l, other.r)

    def __hash__(self):
        return hash(("Add", self.l, self.r))


class Mul(BinOpBase):
    def __repr__(self):
        return f"Mul({self.l}, {self.r})"

    def __eq__(self, other):
        if not isinstance(other, Mul):
            return False

        return (self.l, self.r) == (other.l, other.r)

    def __hash__(self):
        return hash(("Mul", self.l, self.r))


def rewrite_expr(expr: Expr):
    # leaf
    if len(expr.args) == 0:
        return expr

    if expr.func == sympy.core.add.Add and len(expr.args) > 2:
        return sympy.core.add.Add(
            rewrite_expr(expr.args[0]),
            rewrite_expr(sympy.core.add.Add(*expr.args[1:])),
            evaluate=False,
        )
    if expr.func == sympy.core.mul.Mul and len(expr.args) > 2:
        return sympy.core.mul.Mul(
            rewrite_expr(expr.args[0]),
            rewrite_expr(sympy.core.mul.Mul(*expr.args[1:])),
            evaluate=False,
        )

    return expr.func(*[rewrite_expr(arg) for arg in expr.args])


def decompose_goal(expr: Union[Expr, List[Expr]]):
    if isinstance(expr, list):
        for child in expr:
            yield from decompose_goal(child)
    elif expr.func == sympy.core.add.Add and len(expr.args) > 2:
        # Incomplete Hack, in theory should allow for many trees of addition
        # Also, this could become an invalid approximation if SIMD allows for adding, say 4x at a time
        yield from decompose_goal(rewrite_expr(expr))
    elif expr.func == sympy.core.mul.Mul and len(expr.args) > 2:
        # Incomplete Hack, in theory should allow for many trees of multiplication
        # Also, this could become an invalid approximation if SIMD allows for multiplying, say 4x at a time
        yield from decompose_goal(rewrite_expr(expr))
    else:
        for child in expr.args:
            yield from decompose_goal(child)

        yield expr


class CpuState(StateBase):
    @classmethod
    def Start(cls, free_symbols, goal_exprs):
        pipeline = (Nop(), Nop(), Nop())

        goal_exprs = list(decompose_goal(goal_exprs))

        return cls(
            cycle_count=0,
            pipeline=pipeline,
            free_symbols=free_symbols,
            goal_exprs=goal_exprs,
        )

    def __init__(
        self,
        cycle_count: int,
        pipeline: List[InstructionBase],
        free_symbols: Iterable[Any],
        goal_exprs: List[Expr],
    ):
        self.instruction_set = (Add, Mul, Nop)

        self.cycle_count = cycle_count

        self.pipeline = pipeline
        self.computed_values = frozenset(free_symbols)
        self.goal_exprs = frozenset(goal_exprs) - self.computed_values

    def __repr__(self):
        return f"CpuState(pipline={self.pipeline},computed={self.computed_values})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return (self.pipeline, self.computed_values, self.goal_exprs,) == (
            other.pipeline,
            other.computed_values,
            other.goal_exprs,
        )

    def __hash__(self):
        return hash((self.pipeline, self.computed_values))

    def cycle(self, instruction):
        cycle_count = self.cycle_count + 1

        result, next_pipeline = self.pipeline[0], self.pipeline[1:] + (instruction,)

        if isinstance(result, Nop):
            newly_computed = set()
        elif isinstance(result, Add):
            newly_computed = set([result.l + result.r])
        elif isinstance(result, Mul):
            newly_computed = set([result.l * result.r])
        else:
            raise TypeError(f"Misconfigured Instruction Type {type(result)}")

        next_computed = self.computed_values.union(newly_computed)
        next_goal = self.goal_exprs.difference(newly_computed)

        assert next_computed is not None
        return CpuState(
            cycle_count=cycle_count,
            pipeline=next_pipeline,
            free_symbols=next_computed,
            goal_exprs=next_goal,
        )


class InstructionAction(ActionBase):
    def __init__(self, instruction: InstructionBase, cost: float = 1.0):
        self.instruction = instruction
        self.cost = cost

    def __eq__(self, other: ActionBase):
        if not isinstance(other, type(self)):
            return False

        return (self.instruction, self.cost) == (
            other.instruction,
            other.cost,
        )

    def __repr__(self):
        return f"I_A_({self.instruction}, {self.cost})"


class GoalMultiState:
    def __init__(self, target_exprs):
        self.target_exprs = target_exprs

    def __eq__(self, other):
        if not isinstance(other, CpuState):
            return False

        if all(t in other.computed_values for t in self.target_exprs):
            return True

        return False


def heuristic(state: CpuState):
    return len(state.goal_exprs)


def available(expr: Expr, computed_values: Iterable[Expr]) -> bool:
    if expr.func == sympy.core.numbers.NegativeOne:
        return True

    return all(v in computed_values for v in expr.args)


def available_edges(state: CpuState):
    for expr in sorted(list(state.goal_exprs), key=lambda expr: str(expr)):
        if available(expr, state.computed_values):

            if expr.func == sympy.core.add.Add:
                if len(expr.args) != 2:
                    raise ValueError(f"Expression with args {len(expr.args)}: {expr}")
                l, r = expr.args

                candidate = InstructionAction(Add(l, r))

                if candidate.instruction not in state.pipeline:
                    yield candidate
            elif expr.func == sympy.core.mul.Mul:
                if len(expr.args) != 2:
                    raise ValueError(f"Expression with args {len(expr.args)}: {expr}")
                l, r = expr.args

                candidate = InstructionAction(Mul(l, r))

                if candidate.instruction not in state.pipeline:
                    yield candidate
            else:
                print(expr.func)
                1 / 0

    if state.pipeline != (Nop(), Nop(), Nop()):
        yield InstructionAction(instruction=Nop())


def transition(state: CpuState, action: InstructionAction):
    return state.cycle(action.instruction)


def superoptimizer(expressions: List[expr], free_symbols: List[Any]):
    assert (Nop(), Nop(), Nop()) == (Nop(), Nop(), Nop())

    assert len(expressions) >= 1
    assert len(free_symbols) >= 1
    free_symbols += [0, 1, -1]

    initial_state = CpuState.Start(free_symbols=free_symbols, goal_exprs=expressions)
    goal_state = GoalMultiState(expressions)

    instruction_seequence = astar(
        initial_state=initial_state,
        goal_state=goal_state,
        transition=transition,
        heuristic=heuristic,
        edges=available_edges,
    )

    stats = {
        "cycle count": len(instruction_seequence),
        "full": [a.instruction for a in instruction_seequence],
    }

    operations = [
        a.instruction for a in instruction_seequence if a.instruction != Nop()
    ]

    return operations, stats
