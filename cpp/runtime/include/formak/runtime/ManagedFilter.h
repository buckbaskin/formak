#include <any>
#include <vector>

namespace formak::runtime {

template <typename Impl>
class ManagedFilter {
 public:
  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control) {
    return processUpdate(outputTime, control);
  }
  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control,
      const std::vector<typename Impl::StampedReadingT>& readings) {
    for (const auto& stampedReading : readings) {
      _state = processUpdate(stampedReading.timestamp, control);
      _currentTime = stampedReading.timestamp;

      _state = stampedReading.sensor_model(_impl, _state);
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
