from common import ui_model
from formak.backend_cpp import cpp
from formak.backend_cpp.config import Config


def main():
    cpp.compile(ui_model(), config=Config(common_subexpression_elimination=True))


if __name__ == "__main__":
    main()
