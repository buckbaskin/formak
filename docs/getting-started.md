# Getting Started

## Installation

This project uses Bazel as its build system. To get started, make sure you have
Bazelisk, Python3 and Clang available.

### Requirements

- Bazel
- Clang-12 / C++17
- Python3


### Set up Bazelisk

### Install Clang

## Running Some Code

To get started running code for the project, try the command

`make ci`

This will run all of the unit tests for the project and if it passes it indicates that the project is set up correctly

### Common Issues

...

### Next Steps

Using bazel you can specify a more fine-grained set of code to run. For example, if you're interested in the compilation feature available in Python generation, you can run the command

`bazel test //featuretests:python-library-for-model-evaluation`
