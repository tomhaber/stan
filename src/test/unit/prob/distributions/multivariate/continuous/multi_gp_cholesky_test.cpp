#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_gp_cholesky.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsMultiGPCholesky,MultiGPCholesky) {
  Matrix<double,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  double lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<double,Dynamic,1> cy(y.row(i).transpose());
    Matrix<double,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref, stan::prob::multi_gp_cholesky_log(y,L,w));
}

TEST(ProbDistributionsMultiGPCholesky,ErrorL) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  // TODO
}

TEST(ProbDistributionsMultiGPCholesky,ErrorW) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  // negative w
  w(0, 0) = -2.5;
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);

  // non-finite values
  w(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
  w(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
  w(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
}

TEST(ProbDistributionsMultiGPCholesky,ErrorY) {
  Matrix<double,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;
  
  Matrix<double,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
  -3.0,  4.0, 0.0,  0.0, 0.0,
  0.0,  0.0, 5.0,  1.0, 0.0,
  0.0,  0.0, 1.0, 10.0, 0.0,
  0.0,  0.0, 0.0,  0.0, 2.0;
  
  Matrix<double,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  Matrix<double,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;
  
  // non-finite values
  y(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
  y(0, 0) = -std::numeric_limits<double>::infinity();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
  y(0,0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW (stan::prob::multi_gp_cholesky_log(y, L, w), std::domain_error);
}


TEST(ProbDistributionsMultiGPCholesky,fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<double>,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<double>,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<double>,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
    -3.0,  4.0, 0.0,  0.0, 0.0,
    0.0,  0.0, 5.0,  1.0, 0.0,
    0.0,  0.0, 1.0, 10.0, 0.0,
    0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<double>,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;

  for (int i = 0; i < 5; i++) {
    mu(i).d_ = 1.0;
    if (i < 3)
      w(i).d_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_ = 1.0;
      if (i < 3)
        y(i,j).d_ = 1.0;
    }
  }

  Matrix<fvar<double>,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  fvar<double> lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<double>,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<double>,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_, stan::prob::multi_gp_cholesky_log(y,L,w).val_);
  EXPECT_FLOAT_EQ(-74.572952, stan::prob::multi_gp_cholesky_log(y,L,w).d_);
}

TEST(ProbDistributionsMultiGPCholesky,fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  Matrix<fvar<var>,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<var>,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
    -3.0,  4.0, 0.0,  0.0, 0.0,
    0.0,  0.0, 5.0,  1.0, 0.0,
    0.0,  0.0, 1.0, 10.0, 0.0,
    0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<var>,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;

  for (int i = 0; i < 5; i++) {
    mu(i).d_ = 1.0;
    if (i < 3)
      w(i).d_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_ = 1.0;
      if (i < 3)
        y(i,j).d_ = 1.0;
    }
  }

  Matrix<fvar<var>,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  fvar<var> lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<var>,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<var>,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_.val(), stan::prob::multi_gp_cholesky_log(y,L,w).val_.val());
  EXPECT_FLOAT_EQ(-74.572952, stan::prob::multi_gp_cholesky_log(y,L,w).d_.val());
}

TEST(ProbDistributionsMultiGPCholesky,fvar_fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
    -3.0,  4.0, 0.0,  0.0, 0.0,
    0.0,  0.0, 5.0,  1.0, 0.0,
    0.0,  0.0, 1.0, 10.0, 0.0,
    0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<fvar<double> >,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;

  for (int i = 0; i < 5; i++) {
    mu(i).d_.val_ = 1.0;
    if (i < 3)
      w(i).d_.val_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_.val_ = 1.0;
      if (i < 3)
        y(i,j).d_.val_ = 1.0;
    }
  }

  Matrix<fvar<fvar<double> >,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  fvar<fvar<double> > lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<fvar<double> >,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<fvar<double> >,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_.val_, stan::prob::multi_gp_cholesky_log(y,L,w).val_.val_);
  EXPECT_FLOAT_EQ(-74.572952, stan::prob::multi_gp_cholesky_log(y,L,w).d_.val_);
}

TEST(ProbDistributionsMultiGPCholesky,fvar_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  Matrix<fvar<fvar<var> >,Dynamic,1> mu(5,1);
  mu.setZero();
  
  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0, 11.0, 2.0, -5.0, 11.0, 0.0, -2.0, 11.0, 2.0, -2.0, -11.0;

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
    -3.0,  4.0, 0.0,  0.0, 0.0,
    0.0,  0.0, 5.0,  1.0, 0.0,
    0.0,  0.0, 1.0, 10.0, 0.0,
    0.0,  0.0, 0.0,  0.0, 2.0;

  Matrix<fvar<fvar<var> >,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 1.5;

  for (int i = 0; i < 5; i++) {
    mu(i).d_.val_ = 1.0;
    if (i < 3)
      w(i).d_.val_ = 1.0;
    for (int j = 0; j < 5; j++) {
      Sigma(i,j).d_.val_ = 1.0;
      if (i < 3)
        y(i,j).d_.val_ = 1.0;
    }
  }

  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> L = Sigma.llt().matrixL();

  fvar<fvar<var> > lp_ref(0);
  for (size_t i = 0; i < 3; i++) {
    Matrix<fvar<fvar<var> >,Dynamic,1> cy(y.row(i).transpose());
    Matrix<fvar<fvar<var> >,Dynamic,Dynamic> cSigma((1.0/w[i])*Sigma);
    lp_ref += stan::prob::multi_normal_log(cy,mu,cSigma);
  }
  
  EXPECT_FLOAT_EQ(lp_ref.val_.val_.val(), stan::prob::multi_gp_cholesky_log(y,L,w).val_.val_.val());
  EXPECT_FLOAT_EQ(-74.572952, stan::prob::multi_gp_cholesky_log(y,L,w).d_.val_.val());
}
