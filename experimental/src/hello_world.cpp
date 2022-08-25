#include <experimental/hello_greet.h>

#include <iostream>

int main() {
  std::cout << "Hello "
            << "World " << magic() << std::endl;
  return 0;
}
