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

        # print("Debug", "edges", list(edges(state)))

        for action in edges(state):
            next_state = transition(state, action)
            # print("Debug", "transition", state, "->", next_state)

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


def decompose_goal(expr: Expr):
    for child in expr.args:
        yield from decompose_goal(child)

    yield expr


class CpuState(StateBase):
    @classmethod
    def Start(cls, free_symbols, goal_expr):
        pipeline = (Nop(), Nop(), Nop())

        goal_exprs = list(decompose_goal(goal_expr))
        print("Goal", goal_expr)
        print("Decomposed")
        print(goal_exprs)

        return cls(pipeline, free_symbols, goal_exprs)

    def __init__(
        self,
        pipeline: List[InstructionBase],
        free_symbols: Iterable[Any],
        goal_exprs: List[Expr],
    ):
        self.instruction_set = (Add, Mul, Nop)

        self.cycle_count = 0

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
        self.cycle_count += 1

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
        return CpuState(next_pipeline, next_computed, next_goal)


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
        return f"InstructionAction({self.instruction}, {self.cost})"


class GoalMultiState:
    def __init__(self, target_expr):
        self.target_expr = target_expr

    def __eq__(self, other):
        print("GoalMultiState __eq__")
        if not isinstance(other, CpuState):
            return False

        if self.target_expr in other.computed_values:
            return True

        return False


def heuristic(state: CpuState):
    return len(state.goal_exprs)


def available(expr, computed_values):
    if len(expr.args) != 2:
        raise ValueError(f"Expression with args {len(expr.args)}: {expr}")

    l, r = expr.args

    return l in computed_values and r in computed_values


def available_edges(state: CpuState):
    for expr in state.goal_exprs:
        if available(expr, state.computed_values):

            if expr.func == sympy.core.add.Add:
                if len(expr.args) != 2:
                    raise ValueError(f"Expression with args {len(expr.args)}: {expr}")
                l, r = expr.args

                candidate = InstructionAction(Add(l, r))

                if candidate.instruction not in state.pipeline:
                    yield candidate
            else:
                print(expr.func)
                1 / 0

    if state.pipeline != (Nop(), Nop(), Nop()):
        yield InstructionAction(instruction=Nop())


def transition(state: CpuState, action: InstructionAction):
    return state.cycle(action.instruction)


def superoptimizer(expression, free_symbols):
    assert (Nop(), Nop(), Nop()) == (Nop(), Nop(), Nop())
    initial_state = CpuState.Start(free_symbols, expression)
    goal_state = GoalMultiState(expression)

    instruction_seequence = astar(
        initial_state=initial_state,
        goal_state=goal_state,
        transition=transition,
        heuristic=heuristic,
        edges=available_edges,
    )
    print("superoptimizer")
    print("result")
    print(instruction_seequence)
    1 / 0
