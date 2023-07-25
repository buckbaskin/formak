#include <array>
#include <iostream>
#include <vector>

namespace test::tools {

enum class ShuffleId {
  Zero,
  Sensor0,
  Sensor1,
  Sensor2,
  Sensor3,
  Output,
};
std::ostream& operator<<(std::ostream& o, ShuffleId id);
std::ostream& operator<<(std::ostream& o, const std::vector<ShuffleId>& v);

struct OrderOptions {
  double output_dt = 0.0;
  std::array<double, 4> sensor_dt{0.0, 0.0, 0.0, 0.0};
};
std::ostream& operator<<(std::ostream& o, const std::array<double, 4>& a);
std::ostream& operator<<(std::ostream& o, const OrderOptions& options);

std::vector<std::vector<ShuffleId>> AllOrderings(
    const std::vector<ShuffleId>& base_set);

std::vector<OrderOptions> AllOrderOptions();

}  // namespace test::tools
