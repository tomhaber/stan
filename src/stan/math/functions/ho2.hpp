#ifndef __STAN__MATH__HO2_HPP__
#define __STAN__MATH__HO2_HPP__

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/var.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace agrad {
    stan::agrad::var max(stan::agrad::var a, stan::agrad::var b) {
      return fmax(a, b);
    }
  }
}


namespace stan {
  namespace math {
    
    class harm_osc_coupled {
      double gamma_;
  
    public:
      harm_osc_coupled( double gamma ) : gamma_(gamma) { }

      void operator() ( const std::vector<double> &x , 
                        std::vector<double> &dxdt, 
                        const double /* t */ ) {
        dxdt[0] = x[1];
        dxdt[1] = -x[0] - gamma_*x[1];
        dxdt[2] = x[3];
        dxdt[3] = -x[2] - gamma_ * x[3] - x[1];
      }
    };
    

    template<class T>
    struct push_back_state_and_time {
      std::vector< std::vector<T> >& m_states;
      std::vector< T >& m_times;
      
      push_back_state_and_time( std::vector< std::vector<T> > &states , std::vector< T > &times )
        : m_states( states ) , m_times( times ) { }
      
      void operator()( const std::vector<T> &x , T t ) {
        m_states.push_back( x );
        m_times.push_back( t );
      }
      
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
      get() {
        size_t N = m_states.size();
        size_t M = m_states[0].size();
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> xhat(N-1, M);
        
        for (size_t n = 1; n < N; n++) 
          for (size_t m = 0; m < M; m++)
            xhat(n-1,m) = m_states[n][m];
        
        return xhat;
      }
      
      void print() {
        std::cout << "time,x_0";
        for (size_t n = 1; n < m_states[0].size(); n++)
          std::cout << ",x_" << n;
        std::cout << std::endl;
        for (size_t n = 0; n < m_states.size(); n++) {
          std::cout << m_times[n]
                    << "," << m_states[n][0]
                    << "," << m_states[n][1]
                    << "," << m_states[n][2]
                    << "," << m_states[n][3]
                    << std::endl;
        }
      }
    };
    
    Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic> 
    ho2(const std::vector<double>& t,
        const Eigen::Matrix<double,Eigen::Dynamic,1>& x0,
        const stan::agrad::var& gamma) {
      using namespace std;
      using namespace boost::numeric::odeint;
  
      vector<double> x0_state(x0.size()*2);
      for (size_t n = 0; n < x0.size(); n++)
        x0_state[n] = x0[n];
      for (size_t n = x0.size(); n < 2*x0.size(); n++)
        x0_state[n] = 0.0;

      std::vector<double> times(t.size()+1);
      times[0] = 0.0;
      for (size_t n = 0; n < t.size(); n++)
        times[n+1] = t[n];
      
      double dt = times[1] - times[0];

      vector<vector<double> > x_vec;
      vector<double> t_vec;
      push_back_state_and_time<double> obs(x_vec, t_vec);

      harm_osc_coupled harm_osc_coupled(gamma.val());


      integrate_times(make_dense_output(1.0e-6, 1.0e-6, 
                                        runge_kutta_dopri5< vector<double>,
                                                            double,
                                                            vector<double>,
                                                            double>() ) , 
                      harm_osc_coupled, x0_state, 
                      boost::begin(times), boost::end(times), dt, obs);

      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x = obs.get();

      Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic> x_return(t.size(),x0.size());
      for (int n = 0; n < t.size(); n++) 
        for (int m = 0; m < x0.size(); m++) {
          x_return(n,m) 
            = stan::agrad::var(new stan::agrad::precomp_v_vari(x(n,m), gamma.vi_, x(n,m+x0.size())));
        }
      return x_return;
    }
    
  }
}

#endif
