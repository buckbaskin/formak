from common_subexpression_elimination.common import ui_model

from formak import cpp


def main():
    cpp.compile(ui_model(), config=cpp.Config(common_subexpression_elimination=True))


if __name__ == "__main__":
    main()
