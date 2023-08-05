#include <formak/utils/type_checks.h>
#include <gtest/gtest.h>

namespace unit::default_constructable {
using formak::utils::type_checks::DefaultConstructable;

TEST(TypeChecks, DefaultConstructor) {
  struct ImplicitDefaultConstructor {};
  static_assert(DefaultConstructable<ImplicitDefaultConstructor>::value);

  struct ExplicitDefaultConstructor {
    ExplicitDefaultConstructor() = default;
  };
  static_assert(DefaultConstructable<ExplicitDefaultConstructor>::value);

  struct NoDefaultConstructor {
    NoDefaultConstructor() = delete;
  };
  static_assert(!DefaultConstructable<NoDefaultConstructor>::value);
}

}  // namespace unit::default_constructable
