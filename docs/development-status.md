# Development Status

Alpha! FormaK key functionality is under active development.

The intended user experience is as follows. The user provides:
- Model that describes the physics of the system
- Execution criteria (e.g. memory usage, execution time)
- Time series data for the system

The Five Key Elements the library provides to achieve this user experience are:
1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. ([In Progress](https://github.com/buckbaskin/formak/pull/3)) Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. (Planned) C++ and Python to C++ interoperability for performance
5. (Planned) C++ interfaces to support a variety of model uses
