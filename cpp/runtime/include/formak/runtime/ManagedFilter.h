#include <any>
#include <cmath>
#include <memory>
#include <vector>

namespace formak::runtime {

template <typename Impl>
class ManagedFilter {
 public:
  ManagedFilter(double initialTimestamp,
                const typename Impl::StateAndVarianceT& initialState)
      : _impl(),
        _state{.currentTime = initialTimestamp, .state = initialState} {
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
    return processUpdate(outputTime, control).state;
  }
  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control,
      const std::vector<StampedReading>& readings) {
    for (const auto& stampedReading : readings) {
      _state = processUpdate(stampedReading.timestamp, control);

      // No change in time for sensor readings
      _state.state = stampedReading.data->sensor_model(_impl, _state.state);
    }

    return tick(outputTime, control);
  }

 private:
  struct State {
    double currentTime = 0.0;
    typename Impl::StateAndVarianceT state;
  };

  State processUpdate(double outputTime,
                      const typename Impl::ControlT& control) const {
    double dt = 0.1;
    if (_state.currentTime < outputTime) {
      dt *= -1;
    }

    typename Impl::StateAndVarianceT state = _state.state;

    size_t expected_iterations = static_cast<size_t>(
        std::abs(std::floor((outputTime - _state.currentTime) / dt)));

    for (size_t count = 0; count < expected_iterations; ++count) {
      state = _impl.process_model(dt, state, control);
    }
    double iterTime = _state.currentTime + dt * expected_iterations;
    if (std::abs(outputTime - iterTime) >= 1e-9) {
      state = _impl.process_model(outputTime - iterTime, state, control);
    }

    return {.currentTime = outputTime, .state = state};
  }

  const Impl _impl;
  State _state;
};
}  // namespace formak::runtime
