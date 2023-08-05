#include <type_traits>

namespace formak::utils::type_checks {

template <typename T, typename = decltype(T())>
class DefaultConstructable : public std::true_type {};
template <typename T>
class DefaultConstructable<T, std::false_type> : public std::false_type {};

}  // namespace formak::utils::type_checks
