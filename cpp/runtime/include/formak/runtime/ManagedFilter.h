#include <vector>

namespace formak::runtime {
template <typename Impl>
class ManagedFilter {
 public:
  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control) {
    return processUpdate(outputTime, control);
  }
  template <typename ReadingT>
  typename Impl::StateAndVarianceT tick(double outputTime,
                                        const typename Impl::ControlT& control,
                                        const std::vector<ReadingT>& readings) {
    for (const auto& sensorReading : readings) {
      _state = processUpdate(sensorReading.time, _state);
      _currentTime = sensorReading.time;

      _state = _impl.sensor_model(_state, sensorReading);
    }

    return tick(outputTime, control);
  }

 private:
  typename Impl::StateAndVarianceT processUpdate(
      double outputTime, const typename Impl::ControlT& control) const {
    double dt = 0.1;
    typename Impl::StateAndVarianceT state = _state;
    for (double currentTime = _currentTime; currentTime < outputTime;
         currentTime += dt) {
      state = _impl.process_model(dt, state, control);
    }
    return state;
  }

  double _currentTime = 0.0;
  const Impl _impl;
  typename Impl::StateAndVarianceT _state;
};
}  // namespace formak::runtime
