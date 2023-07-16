#include <vector>

namespace formak::runtime {
template <typename Impl>
class ManagedFilter {
 public:
  template <typename ReadingT>
  typename Impl::StateAndVariance tick(double outputTime,
                                       const typename Impl::Control& control,
                                       const std::vector<ReadingT>& readings) {
    for (const auto& sensorReading : readings) {
      _state = processUpdate(sensorReading.time, _state);
      _currentTime = sensorReading.time;

      _state = Impl::sensor_model(_state, sensorReading);
    }

    return Impl::processUpdate(_state, outputTime);
  }

  typename Impl::StateAndVariance processUpdate(
      double outputTime, const typename Impl::Control& control) const {
    double dt = 0.1;
    typename Impl::StateAndVariance state = _state;
    for (double currentTime = _currentTime; currentTime < outputTime;
         currentTime += dt) {
      state = Impl::process_model(dt, state, control);
    }
    return state;
  }

 private:
  double _currentTime = 0.0;
  typename Impl::StateAndVariance _state;
};
}  // namespace formak::runtime
