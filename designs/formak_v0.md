# FormaK V0

Author: Buck Baskin @bebaskin

## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling with code generation to create performant code that is easy to use.

The values (in order) are:
- Easy to use
- Performant

## Solution Approach

Combine tooling and libraries in Python:
- sympy
- scikit-learn
- hypothesis
- etc

with libraries and benefits of C++:
- performant, only pay for what you use
- Eigen
- etc

and end up with an easy to use interface so it's easy to adopt and use.
Working together, we can build "Middle-Out". On the frontend, we can add features to the Python interface. On the backend, we augment the features with performant implementations.

- Python, and specifically sympy, is the primary interface for interacting with the library with a focus on ease of use.
- C++ is a "backend" that executes the model defined in Python with a focus on performance.

### Milestones

- [ ] Python UI Demo
- [ ] C++ Generation Demo
- [ ] Automated CI
- [ ] Automated Feature Test Reporting
- [ ] Writeup of the demo

## Development Process

The development process for adding features will be:

1. Design Doc
2. Write feature tests. When the feature tests pass, the feature is nominally working at an alpha level
3. Implement the feature, including additional testing
4. Code Review
5. Merge via PR

