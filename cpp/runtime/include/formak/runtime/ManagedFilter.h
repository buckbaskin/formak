#include <any>
#include <chrono>  // nanoseconds
#include <cmath>   // floor
#include <iostream>
#include <memory>
#include <type_traits>  // enable_if
#include <vector>

namespace formak::runtime {

class ScopeTimer {
 public:
  ScopeTimer(std::vector<std::chrono::nanoseconds>* record)
      : _record(record), _start(std::chrono::steady_clock::now()) {
    std::cout << "Within ScopeTimer()" << std::endl;
  }
  ~ScopeTimer() {
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    _record->emplace_back(end - _start);
    std::cout << "Within ~ScopeTimer()" << std::endl;
  }

 private:
  std::vector<std::chrono::nanoseconds>* _record;
  std::chrono::steady_clock::time_point _start;
};

template <typename Impl>
class ManagedFilter {
 public:
  // ImplT required here so that the substitution will fail (SFINAE) for this
  // constructor specifically. If we don't substitute it and instead refer Impl,
  // the std::enable_if call will lead to unintended behavior. The ManagedFilter
  // tests with/without Calibration and with/without Control ensure this is
  // working at compile time.
  template <typename EnableT = Impl,
            typename = typename std::enable_if<std::is_same_v<
                typename EnableT::Tag::CalibrationT, std::false_type>>::type>
  ManagedFilter(double initialTimestamp,
                const typename Impl::Tag::StateAndVarianceT& initialState)
      : _impl(),
        _state{.currentTime = initialTimestamp, .state = initialState} {
    static_assert(
        std::is_same_v<typename Impl::Tag::CalibrationT, std::false_type>);
  }
  template <typename EnableT = Impl,
            typename = typename std::enable_if<!std::is_same_v<
                typename EnableT::Tag::CalibrationT, std::false_type>>::type>
  ManagedFilter(double initialTimestamp,
                const typename Impl::Tag::StateAndVarianceT& initialState,
                const typename Impl::Tag::CalibrationT& calibration)
      : _impl(),
        _calibration(calibration),
        _state{
            .currentTime = initialTimestamp,
            .state = initialState,
        } {
    static_assert(
        !std::is_same_v<typename Impl::Tag::CalibrationT, std::false_type>);
  }

  struct StampedReading {
    double timestamp = 0.0;
    std::shared_ptr<typename Impl::Tag::StampedReadingBaseT> data;
  };
  template <typename ReadingT>
  static StampedReading wrap(double timestamp, ReadingT&& reading) {
    return StampedReading{
        .timestamp = timestamp,
        .data = std::shared_ptr<typename Impl::Tag::StampedReadingBaseT>(
            new std::remove_reference_t<ReadingT>(reading)),
    };
  }

  typename Impl::Tag::StateAndVarianceT tick(
      double outputTime, const typename Impl::Tag::ControlT& control) {
    static_assert(
        !std::is_same_v<typename Impl::Tag::ControlT, std::false_type>);
    ScopeTimer s(&_timeLog.tickTimeControl);

    return processUpdate(outputTime, control).state;
  }
  typename Impl::Tag::StateAndVarianceT tick(double outputTime) {
    static_assert(
        std::is_same_v<typename Impl::Tag::ControlT, std::false_type>);
    ScopeTimer s(&_timeLog.tickTime);

    return processUpdate(outputTime).state;
  }

  typename Impl::Tag::StateAndVarianceT tick(
      double outputTime, const typename Impl::Tag::ControlT& control,
      const std::vector<StampedReading>& readings) {
    static_assert(
        !std::is_same_v<typename Impl::Tag::ControlT, std::false_type>);
    ScopeTimer s(&_timeLog.tickTimeControlReadings);

    for (const auto& stampedReading : readings) {
      _state = processUpdate(stampedReading.timestamp, control);

      // No change in time for sensor readings
      if constexpr (!std::is_same_v<typename Impl::Tag::CalibrationT,
                                    std::false_type>) {
        _state.state = stampedReading.data->sensor_model(_impl, _state.state,
                                                         _calibration);
      } else {
        _state.state = stampedReading.data->sensor_model(_impl, _state.state);
      }
    }
    std::cout << "Within tickTimeControlReadings" << std::endl;

    return tick(outputTime, control);
  }
  typename Impl::Tag::StateAndVarianceT tick(
      double outputTime, const std::vector<StampedReading>& readings) {
    static_assert(
        std::is_same_v<typename Impl::Tag::ControlT, std::false_type>);
    ScopeTimer s(&_timeLog.tickTimeReadings);

    for (const auto& stampedReading : readings) {
      _state = processUpdate(stampedReading.timestamp);

      // No change in time for sensor readings
      if constexpr (!std::is_same_v<typename Impl::Tag::CalibrationT,
                                    std::false_type>) {
        _state.state = stampedReading.data->sensor_model(_impl, _state.state,
                                                         _calibration);
      } else {
        _state.state = stampedReading.data->sensor_model(_impl, _state.state);
      }
    }

    return tick(outputTime);
  }

  struct TimeLog {
    std::vector<std::chrono::nanoseconds> tickTimeControl;
    std::vector<std::chrono::nanoseconds> tickTime;
    std::vector<std::chrono::nanoseconds> tickTimeControlReadings;
    std::vector<std::chrono::nanoseconds> tickTimeReadings;
  };
  TimeLog viewTimeData() {
    return _timeLog;
  }

 private:
  struct State {
    double currentTime = 0.0;
    typename Impl::Tag::StateAndVarianceT state;
  };

  State processUpdate(double outputTime,
                      const typename Impl::Tag::ControlT& control) const {
    const double max_dt = ([outputTime](const State& state) {
      static_assert(
          !std::is_same_v<typename Impl::Tag::ControlT, std::false_type>);
      if (state.currentTime >= outputTime) {
        return Impl::Tag::max_dt_sec;
      }
      return -Impl::Tag::max_dt_sec;
    })(_state);

    typename Impl::Tag::StateAndVarianceT state = _state.state;

    size_t expected_iterations = static_cast<size_t>(
        std::abs(std::floor((outputTime - _state.currentTime) / max_dt)));

    for (size_t count = 0; count < expected_iterations; ++count) {
      if constexpr (!std::is_same_v<typename Impl::Tag::CalibrationT,
                                    std::false_type>) {
        state = _impl.process_model(max_dt, state, _calibration, control);
      } else {
        state = _impl.process_model(max_dt, state, control);
      }
    }
    double iterTime = _state.currentTime + max_dt * expected_iterations;
    if (std::abs(outputTime - iterTime) >= 1e-9) {
      if constexpr (!std::is_same_v<typename Impl::Tag::CalibrationT,
                                    std::false_type>) {
        state = _impl.process_model(outputTime - iterTime, state, _calibration,
                                    control);
      } else {
        state = _impl.process_model(outputTime - iterTime, state, control);
      }
    }

    return {.currentTime = outputTime, .state = state};
  }
  State processUpdate(double outputTime) const {
    static_assert(
        std::is_same_v<typename Impl::Tag::ControlT, std::false_type>);
    const double max_dt = ([outputTime](const State& state) {
      if (state.currentTime >= outputTime) {
        return Impl::Tag::max_dt_sec;
      }
      return -Impl::Tag::max_dt_sec;
    })(_state);

    typename Impl::Tag::StateAndVarianceT state = _state.state;

    size_t expected_iterations = static_cast<size_t>(
        std::abs(std::floor((outputTime - _state.currentTime) / max_dt)));

    for (size_t count = 0; count < expected_iterations; ++count) {
      state = _impl.process_model(max_dt, state, _calibration);
    }
    double iterTime = _state.currentTime + max_dt * expected_iterations;
    if (std::abs(outputTime - iterTime) >= 1e-9) {
      state = _impl.process_model(outputTime - iterTime, state, _calibration);
    }

    return {.currentTime = outputTime, .state = state};
  }

  const Impl _impl;
  const typename Impl::Tag::CalibrationT _calibration{};

  State _state;

  TimeLog _timeLog;
};
}  // namespace formak::runtime
