#include <any>
#include <memory>
#include <vector>

namespace formak::runtime {

template <typename Impl>
class ManagedFilter {
 public:
  struct BoxedStampedReading {
    std::shared_ptr<typename Impl::StampedReadingT> data;
  };
  template <typename ReadingT>
  BoxedStampedReading wrap(const ReadingT& reading) const {
    return BoxedStampedReading{
        .data = std::shared_ptr<typename Impl::StampedReadingT>(
            new ReadingT(reading)),
    };
  }

  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control) {
    return processUpdate(outputTime, control);
  }
  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control,
      const std::vector<BoxedStampedReading>& readings) {
    for (const auto& stampedReading : readings) {
      _state = processUpdate(stampedReading.data->timestamp, control);
      _currentTime = stampedReading.data->timestamp;

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
