from matplotlib import pyplot as plt

from sympy import Symbol


def plot_pair(*, states, expected_states, arglist, x_name, y_name, file_id):
    x = Symbol(x_name)
    y = Symbol(y_name)

    plt.plot(
        states[:, arglist.index(x)],
        states[:, arglist.index(y)],
        label="model",
    )
    plt.plot(
        expected_states[:, arglist.index(x)],
        expected_states[:, arglist.index(y)],
        label="reference",
        alpha=0.5,
    )
    plt.scatter(
        states[:, arglist.index(x)],
        states[:, arglist.index(y)],
        label="model",
    )
    plt.scatter(
        expected_states[:, arglist.index(x)],
        expected_states[:, arglist.index(y)],
        label="reference",
        alpha=0.5,
    )
    plt.axis("equal")
    plt.legend()
    plt.savefig(f"{file_id}.png")
    plt.close()
    print(f"Write Pair {file_id}.png")


def plot_quaternion_timeseries(
    *, times, states, expected_states, arglist, x_name, file_id
):
    print(x_name + "w")
    w = Symbol(x_name + "w")
    x = Symbol(x_name + "x")
    y = Symbol(x_name + "y")
    z = Symbol(x_name + "z")

    for symbol in [w, x, y, z]:
        plt.plot(
            times,
            states[:, arglist.index(symbol)],
            label=f"model {symbol}",
        )
        plt.plot(
            times,
            expected_states[:, arglist.index(symbol)],
            label=f"reference {symbol}",
            alpha=0.5,
        )
    # plt.scatter(
    #     times,
    #     states[:, arglist.index(x)],
    #     label="model",
    # )
    # plt.scatter(
    #     times,
    #     expected_states[:, arglist.index(x)],
    #     label="reference",
    #     alpha=0.5,
    # )
    plt.legend()
    plt.savefig(f"{file_id}.png")
    print(f"Write Timeseries {file_id}.png")
