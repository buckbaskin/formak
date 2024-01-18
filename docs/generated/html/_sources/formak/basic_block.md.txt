# Basic Block

## For Contributors

Conceptually, BasicBlock is the same in Python and C++; however, there's some
mismatch in the two.

Some quirks of Basic Block:
- In C++ the target for the assignment is tracked separately. Right now this exists to track the type of the assignment as it varies and allow for both declaring new variables and assigning to existing variables. In Python the target is zipped after the fact
- In C++ the arguments to the function aren't tracked. This means that missing arguments won't be caught until C++ compile time. This should be pulled forward to catch this error in Sympy instead of waiting until the user is compiling the generated code.
