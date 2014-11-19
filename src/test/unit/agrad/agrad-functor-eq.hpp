#include <gtest/gtest.h>

#include <stan/agrad/autodiff.hpp>

namespace test_unit {

  template <typename F>
  struct unary_vector_fun {
    const F& f_;

    unary_vector_fun(const F& f) 
    : f_(f) { }

    template <typename T>
    T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) const {
      return f_(x[0]);
    }
  };

  template <typename F1, typename F2>
  void agrad_eq_unary_fvar_double(const F1& f1,
                                  const F2& f2,
                                  double x) {
    double f1x;
    double df1_dx;
    stan::agrad::derivative(f1,x,f1x,df1_dx);

    double f2x;
    double df2_dx;
    stan::agrad::derivative(f2,x,f2x,df2_dx);
    
    // test values are the same
    EXPECT_FLOAT_EQ(f1x,f2x);

    // test first-order derivatives are same
    EXPECT_FLOAT_EQ(df1_dx, df2_dx);
  }

  template <typename F1, typename F2>
  void agrad_eq_unary_var(const F1& f1,
                          const F2& f2,
                          double x) {

    unary_vector_fun<F1> fun1(f1);

    Eigen::Matrix<double,Eigen::Dynamic,1> x_vec1(1);
    x_vec1[0] = x;
    
    double f1x;
    Eigen::Matrix<double,Eigen::Dynamic,1> grad_f1x;
    stan::agrad::gradient(fun1,x_vec1,f1x,grad_f1x);
    double df1x_dx = grad_f1x[0];
    
    
    unary_vector_fun<F2> fun2(f2);

    Eigen::Matrix<double,Eigen::Dynamic,1> x_vec2(1);
    x_vec2[0] = x;
    
    double f2x;
    Eigen::Matrix<double,Eigen::Dynamic,1> grad_f2x;
    stan::agrad::gradient(fun2,x_vec2,f2x,grad_f2x);
    double df2x_dx = grad_f2x[0];

    EXPECT_FLOAT_EQ(f1x,f2x);
    EXPECT_FLOAT_EQ(df1x_dx, df2x_dx);
  }


  template <typename F1, typename F2>
  void expect_agrad_eq_unary(const F1& f1,
                             const F2& f2,
                             double x) {

    agrad_eq_unary_fvar_double(f1,f2,x);
    agrad_eq_unary_var(f1,f2,x);
  }

  template <typename E, typename F>
  void expect_throws_unary(const F& f,
                           double x) {
    // var version throws
    EXPECT_THROW(f(stan::agrad::var(x)), std::domain_error);

    // fvar version throws
    EXPECT_THROW(f(stan::agrad::fvar<double>(x)), std::domain_error);
  }

}
