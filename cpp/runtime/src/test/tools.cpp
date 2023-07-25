#include <formak/runtime/test/tools.h>

#include <algorithm>  // copy_if
#include <iterator>   // back_inserter
#include <vector>

namespace test::tools {

std::ostream& operator<<(std::ostream& o, ShuffleId id) {
  o << "ShuffleId::";
  o << ([id]() {
    switch (id) {
      case ShuffleId::Zero: {
        return "Zero";
      }
      case ShuffleId::Output: {
        return "Output";
      }
      case ShuffleId::Sensor0: {
        return "Sensor0";
      }
      case ShuffleId::Sensor1: {
        return "Sensor1";
      }
      case ShuffleId::Sensor2: {
        return "Sensor2";
      }
      case ShuffleId::Sensor3: {
        return "Sensor3";
      }
    }
    return "Unk";
  })();
  return o;
}
std::ostream& operator<<(std::ostream& o, const std::vector<ShuffleId>& v) {
  o << "v[";
  for (ShuffleId id : v) {
    o << id << ", ";
  }
  o << "]v";
  return o;
}

std::ostream& operator<<(std::ostream& o, const std::array<double, 4>& a) {
  o << "a[" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << "]a";
  return o;
}
std::ostream& operator<<(std::ostream& o, const OrderOptions& options) {
  o << "OrderOptions{.output_dt=" << options.output_dt
    << ", .sensor_dt=" << options.sensor_dt << "}";
  return o;
}

std::vector<std::vector<ShuffleId>> AllOrderings(
    const std::vector<ShuffleId>& base_set) {
  if (base_set.size() <= 1) {
    return {base_set};
  }

  std::vector<std::vector<ShuffleId>> result;

  for (ShuffleId lead : base_set) {
    std::vector<ShuffleId> the_rest;
    std::copy_if(base_set.cbegin(), base_set.cend(),
                 std::back_inserter(the_rest),
                 [lead](ShuffleId other) { return other != lead; });

    for (std::vector<ShuffleId> shuffles : AllOrderings(the_rest)) {
      shuffles.insert(shuffles.cbegin(), lead);
      result.push_back(shuffles);
    }
  }

  return result;
}

std::vector<OrderOptions> AllOptions() {
  std::vector<OrderOptions> result;

  result.push_back(OrderOptions{});

  for (const std::vector<ShuffleId>& shuffle :
       AllOrderings({ShuffleId::Zero, ShuffleId::Output, ShuffleId::Sensor0})) {
    OrderOptions o;
    for (size_t i = 0; i < shuffle.size(); ++i) {
      double index_dt = i * 0.1;
      switch (shuffle[i]) {
        case ShuffleId::Zero: {
          o.output_dt -= index_dt;
          for (int j = 0; j < 4; ++j) {
            o.sensor_dt[j] -= index_dt;
          }
          break;
        }
        case ShuffleId::Output: {
          o.output_dt += index_dt;
          break;
        }
        case ShuffleId::Sensor0: {
          o.sensor_dt[0] += index_dt;
          break;
        }
        case ShuffleId::Sensor1: {
          o.sensor_dt[1] += index_dt;
          break;
        }
        case ShuffleId::Sensor2: {
          o.sensor_dt[2] += index_dt;
          break;
        }
        case ShuffleId::Sensor3: {
          o.sensor_dt[3] += index_dt;
          break;
        }
      }
    }
    result.push_back(o);
  }

  return result;
}

}  // namespace test::tools
