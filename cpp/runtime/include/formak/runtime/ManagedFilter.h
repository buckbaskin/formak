#include <any>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace formak::runtime {

template <typename Impl>
class ManagedFilter {
 public:
  ManagedFilter(double initialTimestamp,
                const typename Impl::StateAndVarianceT& initialState)
      : _impl(), _currentTime(initialTimestamp), _state(initialState) {
  }

  struct StampedReading {
    double timestamp = 0.0;
    std::shared_ptr<typename Impl::StampedReadingBaseT> data;
  };
  template <typename ReadingT>
  static StampedReading wrap(double timestamp, const ReadingT& reading) {
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
    if (_currentTime == outputTime) {
    } else if (_currentTime < outputTime) {
      double currentTime = _currentTime;
      size_t expected_iterations =
          static_cast<size_t>(std::ceil((outputTime - currentTime) / dt));
      for (size_t count = 0; count < expected_iterations; ++count) {
        currentTime += dt;
        std::cout << "+" << dt << " " << currentTime << " < " << outputTime
                  << " " << count << " / "
                  << " " << expected_iterations << std::endl;
        state = _impl.process_model(dt, state, control);
      }
      if (currentTime < outputTime) {
        state = _impl.process_model(outputTime - currentTime, state, control);
        currentTime = outputTime;
      }
    } else {  // _currentTime > outputTime
      double currentTime = _currentTime;
      size_t expected_iterations =
          static_cast<size_t>(std::ceil((outputTime - currentTime) / -dt));
      for (size_t count = 0; count < expected_iterations; ++count) {
        currentTime += dt;
        std::cout << "-" << dt << " " << currentTime << " > " << outputTime
                  << " " << count << " / "
                  << " " << expected_iterations << std::endl;
        state = _impl.process_model(-dt, state, control);
      }
      if (currentTime > outputTime) {
        state = _impl.process_model(outputTime - currentTime, state, control);
        currentTime = outputTime;
      }
    }

    return state;
  }

  const Impl _impl;
  double _currentTime = 0.0;
  typename Impl::StateAndVarianceT _state;
};
}  // namespace formak::runtime