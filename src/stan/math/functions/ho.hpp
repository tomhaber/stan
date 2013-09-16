#ifndef __STAN__MATH__HO_HPP__
#define __STAN__MATH__HO_HPP__

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
    
    template <class T>
    class harm_osc {
      T m_gam;
  
    public:
      harm_osc( T gam ) : m_gam(gam) { }

      void operator() ( const std::vector<T> &x , 
                        std::vector<T> &dxdt, 
                        const T /* t */ )
      {
        dxdt[0] = x[1];
        dxdt[1] = -x[0] - m_gam*x[1];
      }
    };
    

    template<class T>
    struct push_back_state_and_time {
      std::vector< std::vector<T> >& m_states;
      std::vector< T >& m_times;
      
      push_back_state_and_time( std::vector< std::vector<T> > &states , std::vector< T > &times )
        : m_states( states ) , m_times( times ) { }
      
      void operator()( const std::vector<T> &x , T t ) {
        //std::cout << t << ": (" << x[0] << ", " << x[1] << ")" << std::endl;
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
                    << std::endl;
        }
      }
    };
    
    template <class T, class T0>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    ho(const std::vector<double>& t,
       const Eigen::Matrix<T0,Eigen::Dynamic,1>& x0,
       const T& gamma) {
      using namespace std;
      using namespace boost::numeric::odeint;
  
      vector<T> x0_state(2);
      for (size_t n = 0; n < 2; n++)
        x0_state[n] = x0[n];

      std::vector<T> times(t.size()+1);
      times[0] = 0.0;
      for (size_t n = 0; n < t.size(); n++)
        times[n+1] = t[n];
      
      T times_start = 0.0;
      T times_end = t[t.size()-1];
      T dt = t[0];


      vector<vector<T> > x_vec;
      vector<T> t_vec;
      push_back_state_and_time<T> obs(x_vec, t_vec);

      harm_osc<T> harm_osc(gamma);


      integrate_times(make_dense_output(1.0e-6, 1.0e-6, 
                                        runge_kutta_dopri5< vector<T>,
                                                            T,
                                                            vector<T>,
                                                            T>() ) , 
                      harm_osc, x0_state, boost::begin(times), boost::end(times), dt, obs);
      
      return obs.get();
    }
    
  }
}

#endif
