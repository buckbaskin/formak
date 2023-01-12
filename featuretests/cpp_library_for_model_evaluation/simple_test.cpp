#include <formak/cpp_model.h>  // Generated
#include <gtest/gtest.h>

namespace featuretest {

TEST(CppModel, Simple) {
  formak::State state;
  formak::Control control;

  formak::Model cpp_model;

  formak::State state_next = cpp_model.model(0.1, state, control);
}

}  // namespace featuretest
