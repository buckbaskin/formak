from __future__ import annotations
from collections import namedtuple


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

        if state == goal_state:
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


class Mul(BinOpBase):
    def __repr__(self):
        return f"Mul({self.l}, {self.r})"


class CpuState(StateBase):
    @classmethod
    def Start(cls, free_symbols):
        pipeline = (Nop(), Nop(), Nop())

        return cls(pipeline, free_symbols)

    def __init__(self, pipeline: List[InstructionBase], free_symbols: Iterable[Any]):
        self.instruction_set = (Add, Mul, Nop)

        self.cycle_count = 0

        self.pipeline = pipeline
        self.computed_values = frozenset(free_symbols)

    def __repr__(self):
        return f"CpuState(pipline={self.pipeline},computed={self.computed_values})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self._name == other._name

    def __hash__(self):
        return hash((self.pipeline, self.computed_values))

    def cycle(self, instruction):
        self.cycle_count += 1

        result, next_pipeline = self.pipeline[0], self.pipeline[1:] + (instruction,)

        if isinstance(result, Nop):
            next_computed = self.computed_values
        elif isinstance(result, Add):
            next_computed = self.computed_values.union([result.l + result.r])
        elif isinstance(result, Mul):
            next_computed = self.computed_values.union([result.l * result.r])
        else:
            raise TypeError(f"Misconfigured Instruction Type {type(result)}")

        assert next_computed is not None
        return CpuState(next_pipeline, next_computed)


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
    return 0.0


def available_edges(state: CpuState):
    for InstructionType in state.instruction_set:
        if InstructionType == Nop:
            continue

        for lhs in state.computed_values:
            for rhs in state.computed_values:
                yield InstructionAction(instruction=InstructionType(lhs, rhs))

    if state.pipeline != (Nop(), Nop(), Nop()):
        yield InstructionAction(instruction=Nop())


def transition(state: CpuState, action: InstructionAction):
    return state.cycle(action.instruction)


def superoptimizer(expression, free_symbols):
    assert (Nop(), Nop(), Nop()) == (Nop(), Nop(), Nop())
    initial_state = CpuState.Start(free_symbols)
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
