/// Feature Test
///
/// Generate a Model in C++ and test a model iteration.
/// Passes if the model matches the definition.

#include <formak/cpp-model.h>  // Generated
#include <gtest/gtest.h>

namespace featuretest {

TEST(CppModel, Simple) {
  formak::State state({.v = 1.0});
  formak::Control control;

  formak::Model cpp_model;

  EXPECT_DOUBLE_EQ(state.z(), 0.0);
  formak::State state_next = cpp_model.model(0.1, state, control);
  EXPECT_GT(state_next.z(), 0.0);
}

}  // namespace featuretest
