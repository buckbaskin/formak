#include <any>
#include <vector>

namespace formak::runtime {

template <typename Impl>
class ManagedFilter {
 private:
  struct StampedReading {
    double time;
    virtual typename Impl::StateAndVarianceT sensor_model(
        const Impl& impl,
        const typename Impl::StateAndVarianceT& input_state) const = 0;
  };

 public:
  template <typename ReadingT>
  struct StampedReadingImpl {
    typename Impl::StateAndVarianceT sensor_model(
        const Impl& impl,
        const typename Impl::StateAndVarianceT& input_state) const override {
      return impl.sensor_model(input_state, reading);
    }
    ReadingT reading;
  };

  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control) {
    return processUpdate(outputTime, control);
  }
  typename Impl::StateAndVarianceT tick(
      double outputTime, const typename Impl::ControlT& control,
      const std::vector<StampedReading>& readings) {
    for (const StampedReading& stampedReading : readings) {
      _state = processUpdate(stampedReading.time, control);
      _currentTime = stampedReading.time;

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
