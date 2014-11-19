#include <test/unit/agrad/agrad-functor-eq.hpp>

struct log1p_primitive {
  template <typename T>
  T operator()(const T& x) const {
    using ::log;
    return log(1 + x);
  }
};

struct log1p_stan {
  template <typename T>
  T operator()(const T& x) const {
    using stan::math::log1p;
    return log1p(x);
  }
};

TEST(stanMath, log1p_derivs) {
  using test_unit::expect_agrad_eq_unary;
  log1p_primitive f1;
  log1p_stan f2;
  expect_agrad_eq_unary(f1,f2,0.3);
}

// you would use this if log1p were defined to throw, but it's not

// TEST(stanMath, log1p_throws) {
//   using test_unit::expect_throws_unary;
//   log1p_stan f;

//   expect_throws_unary<std::domain_error>(f,-2.7);
// }
