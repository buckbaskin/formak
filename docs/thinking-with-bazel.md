# Thinking In Bazel

Concept Tiers

## Basics

- build environment
- targets
	- named bazel objects
- tools
	- A subset of targets/special case of targets
- Starlark
	- Language used in BUILD files
- Build rules


## C++
- Build rules
	- cc_toolchain
	- cc_binary

## Building C++

- C++ toolchain

- .bazelrc
	- --config=config-name
	- build:config-name
- --crosstool_top
	- Control the toolchain used to build targets
- --host_crosstool_top
	- Control the toolchain used to build tools used in the toolchain
- --cpu
	- --cpu=k8

## Annotated Bazel Tutorial

Source: [Bazel Tutorial: Configure C++ Toolchains](https://bazel.build/tutorials/ccp-toolchain-config)

> This tutorial uses an example scenario to describe how to configure C++
> toolchains for a project.

The tutorial covers defining the logic used to assemble the tools used to build
C++ executables.

> It's based on an
> [example C++ project](https://github.com/bazelbuild/examples/tree/master/cpp-tutorial/stage1)
> that builds error-free using clang.

### What you'll learn

In this tutorial you learn how to:

- Set up the build environment
- Configure the C++ toolchain
- Create a Starlark rule that provides additional configuration for the cc_toolchain so that Bazel can build the application with clang
- Confirm expected outcome by running bazel build --config=clang_config //main:hello-world on a Linux machine
- Build the C++ application
