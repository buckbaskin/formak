#include <cse/cse-model.h>  // Generated
#include <gtest/gtest.h>
#include <no_cse/no-cse-model.h>  // Generated

#include <random>

namespace featuretest {

TEST(CppModel, Simple) {
  cse::Model cse_model;
  no_cse::Model no_cse_model;

  cse_model.model(0.1, cse::StateOptions{.left = 1.0, .right = 0.1});
  no_cse_model.model(0.1, no_cse::StateOptions{.left = 1.0, .right = 0.1});

  auto input = ([]() {
    size_t seed = 1;
    std::minstd_rand prng(seed);
    std::uniform_real_distribution<> dist(0, 1);

    std::vector<double> result;

    for (int i = 0; i < 20; i++) {
      result.push_back(dist(prng));
    }

    return result;
  })();

  for (double d : input) {
    std::cout << d << std::endl;
  }

  FAIL() << "HCF";
}

}  // namespace featuretest
