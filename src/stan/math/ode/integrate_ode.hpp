#ifndef STAN__MATH__ODE__INTEGRATE_ODE_HPP
#define STAN__MATH__ODE__INTEGRATE_ODE_HPP

#include <ostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/error_handling/scalar/check_less.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/error_handling/matrix/check_ordered.hpp>

#include <stan/math/ode/coupled_ode_system.hpp>
#include <stan/math/ode/coupled_ode_observer.hpp>
#include "CVodeIntegrator.h"

namespace stan {

  namespace math {

		template <typename F, typename T1, typename T2>
		class CVodeODE : public CVodeIntegrator::Ode {
			public:
					CVodeODE(const F& f,
                         const std::vector<T1> & y0,
                         const std::vector<T2> & theta,
                         const std::vector<double> & x,
                         const std::vector<int> & x_int,
                         std::ostream* msgs) : cos(f, y0, theta, x, x_int, msgs) {

						y_t.resize( cos.size() );
						dy_dt.resize( cos.size() );
					}

				public:
					int numberOfEquations() const { return cos.size(); }
					int rhs(const double & t, double y[], double ydot[]) {
						std::copy(&y[0], &y[cos.size()], y_t.begin());
						cos(y_t, dy_dt, t);
						std::copy(dy_dt.begin(), dy_dt.end(), &ydot[0]);
						return 0;
					}

				public:
					inline std::vector<double> initial_state() {
						return cos.initial_state();
					}

					inline std::vector<std::vector<typename stan::return_type<T1,T2>::type> >
					decouple_states(const std::vector<std::vector<double> >& y) {
						return cos.decouple_states(y);
					}

				private:
					coupled_ode_system<F, T1, T2> cos;
					std::vector<double> y_t, dy_dt;
		};

    /**
     * Return the solutions for the specified system of ordinary
     * differential equations given the specified initial state,
     * initial times, times of desired solution, and parameters and
     * data, writing error and warning messages to the specified
     * stream.
     *
     * <b>Warning:</b> If the system of equations is stiff, roughly
     * defined by having varying time scales across dimensions, then
     * this solver is likely to be slow.
     *
     * This function is templated to allow the initial times to be
     * either data or autodiff variables and the parameters to be data
     * or autodiff variables.  The autodiff-based implementation for
     * reverse-mode are defined in namespace <code>stan::agrad</code>
     * and may be invoked via argument-dependent lookup by including
     * their headers.
     *
     * This function uses the <a
     * href="http://en.wikipedia.org/wiki/Dormand–Prince_method">Dormand-Prince
     * method</a> as implemented in Boost's <code>
     * boost::numeric::odeint::runge_kutta_dopri5</code> integrator.
     *
     * @tparam F type of ODE system function.
     * @tparam T1 type of scalars for initial values.
     * @tparam T2 type of scalars for parameters.
     * @param[in] f functor for the base ordinary differential equation.
     * @param[in] y0 initial state.
     * @param[in] t0 initial time.
     * @param[in] ts times of the desired solutions, in strictly
     * increasing order, all greater than the initial time.
     * @param[in] theta parameter vector for the ODE.
     * @param[in] x continuous data vector for the ODE.
     * @param[in] x_int integer data vector for the ODE.
     * @param[in,out] msgs the print stream for warning messages.
     * @return a vector of states, each state being a vector of the
     * same size as the state variable, corresponding to a time in ts.
     */
    template <typename F, typename T1, typename T2>
    std::vector<std::vector<typename stan::return_type<T1,T2>::type> >
    integrate_ode(const F& f,
                  const std::vector<T1> y0,
                  const double t0,
                  const std::vector<double>& ts,
                  const std::vector<T2>& theta,
                  const std::vector<double>& x,
                  const std::vector<int>& x_int,
                  std::ostream* msgs) {            
      using boost::numeric::odeint::integrate_times;  
      using boost::numeric::odeint::make_dense_output;  
      using boost::numeric::odeint::runge_kutta_dopri5;
      
      stan::error_handling::check_finite("integrate_ode", "initial state", y0);
      stan::error_handling::check_finite("integrate_ode", "initial time", t0);
      stan::error_handling::check_finite("integrate_ode", "times", ts);
      stan::error_handling::check_finite("integrate_ode", "parameter vector", theta);
      stan::error_handling::check_finite("integrate_ode", "continuous data", x);

      stan::error_handling::check_nonzero_size("integrate_ode", "times", ts);
      stan::error_handling::check_nonzero_size("integrate_ode", "initial state", y0);
      stan::error_handling::check_ordered("integrate_ode", "times", ts);
      stan::error_handling::check_less("integrate_ode", "initial time", t0, ts[0]);

      const double absolute_tolerance = 1e-12;
      const double relative_tolerance = 1e-9;
      CVodeODE<F, T1, T2> coupled_system(f, y0, theta, x, x_int, msgs);

      std::vector<std::vector<double> > y_coupled(ts.size());

      // the coupled system creates the coupled initial state
      std::vector<double> initial_coupled_state = coupled_system.initial_state();

			CVodeIntegrator integrator;
			integrator.assignOde(coupled_system, t0, initial_coupled_state);
			integrator.setRelativeTolerance(relative_tolerance);
			integrator.setAbsoluteTolerance(absolute_tolerance);

      for (size_t n = 0; n < ts.size(); n++) {
				integrator.integrateTo( ts[n] );
				y_coupled[n] = integrator.currentState();
			}

      // the coupled system also encapsulates the decoupling operation
      return coupled_system.decouple_states(y_coupled);
    }

  }

}

#endif
