from common import ui_model
from formak import cpp


def main():
    cpp.compile(ui_model(), config=cpp.Config(common_subexpression_elimination=False))


if __name__ == "__main__":
    main()
