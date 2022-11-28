from fractions import Fraction
from collections import namedtuple

WideRegister = namedtuple("WideRegister", ["lowest", "low", "high", "highest"])


class Instruction:
    def __init__(self, args):
        self.args = args

    def compute_result(self, available_registers):
        raise NotImplementedError()


class Add(Instruction):
    def __init__(self, left_register_name, right_register_name):
        super().__init__(args=[left_register_name, right_register_name])

        self.left = left_register_name
        self.right = right_register_name

    def compute_result(self, available_registers):
        l = available_registers[self.left]
        r = available_registers[self.right]
        return WideRegister(*[l[idx] + r[idx] for idx in range(4)])


CpuArchitectureInfo = namedtuple("CPU_Architecture_Info", ["latency", "throughput"])

SKYLAKE = CpuArchitectureInfo(latency={Add: 4}, throughput={Add: Fraction(1, 2)})


def initial_state():
    return {"available_registers": {}, "inprogress_registers": {}}


def cpu_tick(state, instructions, cpu_architecture_info):
    assert isinstance(state, dict)

    next_instruction, remaining_instructions = instructions[0], instructions[1:]

    if all(
        (
            register_name in state["available_registers"]
            for register_name in next_instruction.args
        )
    ):
        # Data Available, Process Intruction
        return state, remaining_instructions

    for register_name in next_instruction.args:
        if (
            register_name not in state["available_registers"]
            and register_name not in state["inprogress_registers"]
        ):
            raise ValueError(
                "Instruction %s references register %s which is not in available or inprogress registers. Is there a missing load?"
                % (next_instruction, register_name)
            )

    # Stalled
    return state, instructions


state = initial_state()
instructions = []


def run_instruction_list(
    instructions, state=None, *, cpu_architecture_info=None, max_iterations=1000
):
    assert len(instructions) > 0

    if state is None:
        state = initial_state()

    if cpu_architecture_info is None:
        cpu_architecture_info = SKYLAKE

    cpu_stats = {"ticks": 0}

    for i in range(max_iterations):
        state, remaining_instructions = cpu_tick(
            state, instructions, cpu_architecture_info
        )

        cpu_stats["ticks"] += 1

        if len(remaining_instructions) == 0:
            break


def main():
    instruction_sets = [[Add("A", "B")]]
    for idx, instructions in enumerate(instruction_sets):
        stats = run_instruction_list(instructions, cpu_architecture_info=SKYLAKE)
        print("Stats For Instructions %d:" % (idx,))
        print(stats)


if __name__ == "__main__":
    main()
