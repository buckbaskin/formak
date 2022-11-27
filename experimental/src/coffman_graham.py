import sympy
import logging

from sympy import symbols

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("coffman_graham")
logger.setLevel(logging.DEBUG)


def visit_sympy_expr(expr, matcher, base=None):
    if base is None:
        base = tuple()

    if matcher(expr):
        yield base, expr

    for idx, arg in enumerate(expr.args):
        for result in visit_sympy_expr(arg, matcher, base + (idx,)):
            yield result


def setup_coffman_graham(graph, matcher):
    ordered_nodes = []  # List[Tuple[path, sympy expr]]
    unordered_nodes = {}  # Dict[path, sympy expr]

    for path_to_expr, expr in visit_sympy_expr(graph, matcher=matcher):
        if len(expr.args) == 0:
            ordered_nodes.append((path_to_expr, expr))
        else:
            unordered_nodes[tuple(path_to_expr)] = expr

    return ordered_nodes, unordered_nodes


def is_dependendency(node, maybe_dependency):
    node_path, _ = node
    maybe_dep_path, _ = maybe_dependency
    return maybe_dep_path[: len(node_path)] == node_path


assert is_dependendency(((1,), "expr"), ((1, 2), "expr"))
assert not is_dependendency(((2,), "expr"), ((1, 2), "expr"))
assert not is_dependendency(
    (
        (
            1,
            2,
            3,
        ),
        "expr",
    ),
    ((1, 2), "expr"),
)


def cg(graph, width, matcher=None):
    if matcher is None:
        matcher = lambda x: True

    ordered_nodes, unordered_nodes = setup_coffman_graham(graph, matcher)

    def coffman_graham_filter_criteria(node):
        path_to_expr, expr = node

        for idx in range(len(expr.args)):
            if tuple(path_to_expr + (idx,)) in unordered_nodes:
                # Don't try to insert nodes that depend on unordered nodes
                return False

        # Node only depends on ordered nodes
        return True

    def coffman_graham_sort_criteria(node):
        path_to_expr, expr = node
        args_depth_from_end = []

        for arg_idx, arg in enumerate(expr.args):
            arg_path = path_to_expr + (arg_idx,)
            for ordered_idx, (ordered_path, ordered_expr) in enumerate(
                reversed(ordered_nodes)
            ):
                if arg_path == ordered_path:
                    args_depth_from_end.append(ordered_idx)
                    break

        return sorted(args_depth_from_end)

    while len(unordered_nodes) > 0:
        next_in_order = min(
            filter(coffman_graham_filter_criteria, unordered_nodes.items()),
            key=coffman_graham_sort_criteria,
        )
        path, _expr = next_in_order

        ordered_nodes.append(next_in_order)
        del unordered_nodes[tuple(path)]

    assert len(unordered_nodes) == 0

    levels = []

    for idx, node in enumerate(reversed(ordered_nodes)):
        min_level = -1
        for inv_level_idx, level in enumerate(reversed(levels)):
            logger.debug(
                "checking for dependencies for node %s at level %d with %d elements"
                % (node, inv_level_idx, len(level))
            )
            for maybe_dependency in level:
                if is_dependendency(maybe_dependency, node):
                    logger.debug(
                        "point-to dep %s found for node %s at inverse level %d"
                        % (maybe_dependency, node, inv_level_idx)
                    )
                    min_level = len(levels) - inv_level_idx - 1
                    # need to go to a greater level
                    break
                else:
                    logger.debug(
                        "no dep %s found for node %s at inverse level %d"
                        % (maybe_dependency, node, inv_level_idx)
                    )
            if min_level > -1:
                logger.debug(
                    "end of level scan: point-to deps found for node %s at inverse level %d"
                    % (node, inv_level_idx)
                )
                break
            else:
                logger.debug(
                    "end of level scan: no point-to deps found for node %s at inverse level %d"
                    % (node, inv_level_idx)
                )

        for level_idx in range(min_level + 1, len(levels)):
            if len(levels[level_idx]) < width:
                levels[level_idx].append(node)
                break
            else:
                logger.debug("%d reached width" % (level_idx,))
        else:
            # We didn't find an acceptable level, so put it at a new higher level
            levels.append([node])

        logger.debug("End Iteration %d. Levels:" % (idx,))
        for idx, level in enumerate(levels):
            logger.debug("    %d : %s" % (idx, level))

    return levels


def main():
    a, b, c, d = symbols(["a", "b", "c", "d"])
    example_expr = (a + b) * (c + ((a + b) * d))

    print("Visit the expression")
    for idx, val in enumerate(visit_sympy_expr(example_expr, lambda x: True)):
        print(idx, val)

    levels = cg(example_expr, 4)
    print("End Of Calculation. Levels:")
    for idx, level in enumerate(levels):
        print("    %d : %s" % (idx, level))

    def match_Add(expr):
        return expr.func == sympy.core.add.Add

    levels = cg(example_expr, 4, match_Add)
    print("End Of Calculation. Levels:")
    for idx, level in enumerate(levels):
        print("    %d : %s" % (idx, level))


if __name__ == "__main__":
    main()
