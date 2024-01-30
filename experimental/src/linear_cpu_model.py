import logging
from fractions import Fraction
from collections import namedtuple

logger = logging.getLogger("cpu")

WideRegister = namedtuple("WideRegister", ["lowest", "low", "high", "highest"])


class Instruction:
    def __init__(self, target, args):
        self.target = target
        self.args = args

    def compute_result(self, available_registers):
        raise NotImplementedError()


class Add(Instruction):
    def __init__(self, target_register_name, left_register_name, right_register_name):
        super().__init__(
            target=target_register_name, args=[left_register_name, right_register_name]
        )

        self.left = left_register_name
        self.right = right_register_name

    def compute_result(self, available_registers):
        l = available_registers[self.left]
        r = available_registers[self.right]
        return WideRegister(*[l[idx] + r[idx] for idx in range(4)])


CpuArchitectureInfo = namedtuple("CpuArchitectureInfo", ["latency", "throughput"])

SKYLAKE = CpuArchitectureInfo(latency={Add: 4}, throughput={Add: Fraction(1, 2)})


def initial_state():
    return {"tick_count": -1, "available_registers": {}, "inprogress_registers": {}}


def cpu_tick(state, instructions, cpu_architecture_info):
    assert isinstance(state, dict)

    state["tick_count"] += 1

    # Note: this logic will overwrite a previous "inprogress" write to the same register
    for register_name, (available_tick, register) in list(
        state["inprogress_registers"].items()
    ):
        if available_tick <= state["tick_count"]:
            state["available_registers"][register_name] = register
            del state["inprogress_registers"][register_name]

    if len(instructions) == 0:
        return state, instructions
    next_instruction, remaining_instructions = instructions[0], instructions[1:]

    stalled = False
    for register_name in next_instruction.args:
        if register_name in state["inprogress_registers"]:
            stalled = True
        else:
            if register_name not in state["available_registers"]:
                raise ValueError(
                    "Instruction %s references register %s which is not in available or inprogress registers. Is there a missing load?"
                    % (next_instruction, register_name)
                )

    if stalled:
        logger.debug("Stalled")
        return state, instructions

    # Data Available, Process Intruction
    logger.debug("Processing Instruction {}".format(next_instruction))

    # Note: Ignores throughput (for better and worse)
    state["inprogress_registers"][next_instruction.target] = (
        state["tick_count"] + cpu_architecture_info.latency[type(next_instruction)],
        next_instruction.compute_result(state["available_registers"]),
    )

    return state, remaining_instructions


state = initial_state()
instructions = []  # type: List[Instruction]


def run_instruction_list(
    instructions, *, state=None, cpu_architecture_info=None, max_iterations=1000
):
    assert len(instructions) > 0

    if state is None:
        state = initial_state()

    if cpu_architecture_info is None:
        cpu_architecture_info = SKYLAKE

    cpu_stats = {"ticks": 0, "stalls": 0}

    for i in range(max_iterations):
        len_starting_instructions = len(instructions)

        state, instructions = cpu_tick(state, instructions, cpu_architecture_info)

        len_ending_instructions = len(instructions)
        if len_starting_instructions == len_ending_instructions:
            cpu_stats["stalls"] += 1

        cpu_stats["ticks"] += 1

        if len(instructions) == 0 and len(state["inprogress_registers"]) == 0:
            break
    else:
        raise ValueError(
            "Instructions Reached Maximum Iterations %d without terminating"
            % (max_iterations,)
        )

    return cpu_stats, state


def main():
    instruction_sets = [
        [Add("Z", "A", "B"), Add("Y", "C", "D")],
        [Add("data_dependency", "A", "B"), Add("Y", "C", "data_dependency")],
    ]
    for idx, instructions in enumerate(instruction_sets):
        state = initial_state()
        state["available_registers"]["A"] = WideRegister(1, 3, 0, -1)
        state["available_registers"]["B"] = WideRegister(2, 0, 3, 4)

        state["available_registers"]["C"] = WideRegister(-1, -3, -0, 1)
        state["available_registers"]["D"] = WideRegister(-2, -0, -3, -4)

        stats, state = run_instruction_list(
            instructions, state=state, cpu_architecture_info=SKYLAKE, max_iterations=10
        )
        logger.info("State For Instruction List %d:" % (idx,))
        logger.info(state)
        print("Stats For Instruction List %d:" % (idx,))
        print(stats)


if __name__ == "__main__":
    main()
