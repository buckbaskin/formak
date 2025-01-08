import logging

from backend_cpp.compile_shared import _compile_argparse, _compile_impl
from backend_cpp.config import Config
from backend_cpp.model import Model
from frontend.model_validation import model_validation

DEFAULT_MODULES = ("scipy", "numpy", "math")

logger = logging.getLogger(__name__)


def _generate_model_function_bodies(
    header_location, namespace, symbolic_model, calibration_map, config
):
    # For .../generated/formak/xyz.h
    # I want formak/xyz.h , so strip a leading generated prefix if present
    if header_location is not None and "generated/" in header_location:
        header_include = header_location.split("generated/")[-1]
    else:
        header_include = "generated_to_stdout.h"

    generator = Model(
        symbolic_model, calibration_map, namespace, header_include, config
    )

    return generator


def compile_model(symbolic_model, calibration_map=None, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    args = _compile_argparse()

    if args.header is None:
        logger.warning("No Header specified, so output to stdout")
    generator = _generate_model_function_bodies(
        header_location=args.header,
        namespace=args.namespace,
        symbolic_model=symbolic_model,
        calibration_map=calibration_map,
        config=config,
    )

    return _compile_impl(args, generator=generator)
