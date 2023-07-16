#include <any>
#include <memory>
#include <vector>

namespace formak::runtime {

template <typename Impl>
class ManagedFilter {
 public:
  struct StampedReading {
    double timestamp = 0.0;
    std::shared_ptr<typename Impl::StampedReadingBaseT> data;
  };
  template <typename ReadingT>
  StampedReading wrap(double timestamp, const ReadingT& reading) const {
    return StampedReading{
        .timestamp = timestamp,
        .data = std::shared_ptr<typename Impl::StampedReadingBaseT>(
            new ReadingT(reading)),
    };
  }

  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control) {
    return processUpdate(outputTime, control);
  }
  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control,
      const std::vector<StampedReading>& readings) {
    for (const auto& stampedReading : readings) {
      _state = processUpdate(stampedReading.timestamp, control);
      _currentTime = stampedReading.timestamp;

      _state = stampedReading.data->sensor_model(_impl, _state);
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
