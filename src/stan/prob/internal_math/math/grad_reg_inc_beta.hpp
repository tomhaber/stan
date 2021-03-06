#ifndef STAN__PROB__INTERNAL_MATH__MATH__GRAD_REG_INC_BETA_HPP
#define STAN__PROB__INTERNAL_MATH__MATH__GRAD_REG_INC_BETA_HPP

#include <math.h>

#include <stan/prob/internal_math/math/grad_inc_beta.hpp>
#include <stan/prob/internal_math/math/inc_beta.hpp>
#include <stan/math/functions/lbeta.hpp>

namespace stan {
    
  namespace math {

    // Gradient of the regularized incomplete beta function ibeta(a, b, z)
    template<typename T>
    void grad_reg_inc_beta(T& g1, T& g2, T a, T b, T z, 
                           T digammaA, T digammaB, T digammaSum, T betaAB)
    {
      using stan::math::inc_beta;
      using stan::math::grad_inc_beta;
      using std::exp;
      using stan::math::lbeta;

      T dBda = 0;
      T dBdb = 0;
          
      grad_inc_beta(dBda, dBdb, a, b, z);
          
      T b1 = exp(lbeta(a,b)) * inc_beta(a, b, z);
          
      g1 = ( dBda - b1 * (digammaA - digammaSum) ) / betaAB;
      g2 = ( dBdb - b1 * (digammaB - digammaSum) ) / betaAB;
          
    }
      
      
  }

}

#endif
