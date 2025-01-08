from common import ui_model
from formak.backend_cpp.compile_model import compile_model
from formak.backend_cpp.config import Config


def main():
    compile_model(ui_model(), config=Config(common_subexpression_elimination=False))


if __name__ == "__main__":
    main()
