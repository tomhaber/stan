#ifndef CVODEINTEGRATOR_H
#define CVODEINTEGRATOR_H

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <cvode/cvode_dense.h>

#include <vector>
#include <sstream>
#include <stdexcept>

class CVodeIntegrator {
	public:
		class Ode {
		  public:
			 virtual int numberOfEquations() const = 0;
			 virtual int rhs( const double & t, double y[], double ydot[]) = 0;

			 virtual bool hasJacobian() const { return false; }
			 virtual int jacobian(const double t, const double y[], double *J[]) {
				 throw std::runtime_error("no jacobian defined");
			 }
		};

    public:
        CVodeIntegrator();
        ~CVodeIntegrator();

	public:
        void integrateForwardDt( const double dt );
        void integrateTo( const double tOut );
        void setRelativeTolerance( double relTol );
        void setAbsoluteTolerance( double absTol );
        double relativeTolerance() const;
        double absoluteTolerance() const;

        double currentTime() const;
        const std::vector<double> & currentState() const;

				void assignOde(CVodeIntegrator::Ode & odeProblem, const double initialTime, const std::vector<double> & initialState );

    protected:
        void * cvodeMem_;
        N_Vector nvCurrentState_;

        bool initialized_;
        int maxNumSteps_;
				CVodeIntegrator::Ode     *odeProblem_;
        double         relTol_;
        double         absTol_;
        double         currentTime_;
				std::vector<double> currentState_;

        // no copy constructor
        CVodeIntegrator( const CVodeIntegrator & );
};

inline void CVodeIntegrator::integrateForwardDt( const double dt ) {
    integrateTo( dt + currentTime() );
}

inline double CVodeIntegrator::currentTime() const {
    return currentTime_;
}

inline const std::vector<double> & CVodeIntegrator::currentState() const {
    return currentState_;
}

extern "C"
int OdeRhs( double t, N_Vector y, N_Vector ydot, void * f_data ) {
	CVodeIntegrator::Ode * explicitOde = ( CVodeIntegrator::Ode * ) f_data;
    return explicitOde->rhs( t, NV_DATA_S(y) , NV_DATA_S(ydot) );
}

extern "C"
int OdeJacobian(long int N, realtype t, N_Vector y, N_Vector fy,
		DlsMat Jac, void *jac_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
	CVodeIntegrator::Ode * explicitOde = ( CVodeIntegrator::Ode * ) jac_data;
    int ret = explicitOde->jacobian( t, NV_DATA_S(y), Jac->cols );
	return ret;
}

extern "C"
void SilentErrHandler(int error_code, const char *module, const char *function, char *msg, void *eh_data) {
}

CVodeIntegrator::CVodeIntegrator()
	: maxNumSteps_( 50000 ),
		odeProblem_( 0 ),
		relTol_( 1.0e-6 ),
		absTol_( 1.0e-9 ),
		currentTime_( 0.0 ) {

    cvodeMem_ = CVodeCreate( CV_BDF, CV_NEWTON );
    if( cvodeMem_ == 0 )
	   throw std::runtime_error( "CVodeCreate failed" );

	 CVodeSetErrHandlerFn(cvodeMem_, SilentErrHandler, 0);
}

CVodeIntegrator::~CVodeIntegrator() {
	CVodeFree( &cvodeMem_ );
}

void CVodeIntegrator::integrateTo( const double tOut ) {
	if( tOut == currentTime_ ) return;

	int flag = CVode( cvodeMem_, tOut, nvCurrentState_, &currentTime_, CV_NORMAL );
	if( flag != CV_SUCCESS ) {
		std::ostringstream ss; ss << "Cvode error: " << flag;
		throw std::runtime_error(ss.str());
	}
}

void CVodeIntegrator::setRelativeTolerance( double relTol ) {
    relTol_ = relTol;
    int flag = CVodeSStolerances( cvodeMem_, relTol_, absTol_ );
	if( flag != CV_SUCCESS ) {
		std::ostringstream ss; ss << "Failed to set relative tolerance: " << flag;
		throw std::runtime_error(ss.str());
	}
}

void CVodeIntegrator::setAbsoluteTolerance( double absTol ) {
    absTol_ = absTol;
    int flag = CVodeSStolerances( cvodeMem_, relTol_, absTol_ );
	if( flag != CV_SUCCESS ) {
		std::ostringstream ss; ss << "Failed to set absolute tolerance: " << flag;
		throw std::runtime_error(ss.str());
	}
}

void CVodeIntegrator::assignOde(CVodeIntegrator::Ode & odeProblem, const double initialTime, const std::vector<double> & initialState ) {
	currentTime_ = initialTime;
	currentState_ = initialState;
	odeProblem_ = &odeProblem;

    nvCurrentState_ = N_VMake_Serial( currentState_.size(), &currentState_[0] );

    int flag = CVodeInit( cvodeMem_, OdeRhs, currentTime_, nvCurrentState_ );
	if( flag != CV_SUCCESS ) {
		std::ostringstream ss; ss << "Cvode init error: " << flag;
		throw std::runtime_error(ss.str());
	}

    setRelativeTolerance( relTol_ );
    setAbsoluteTolerance( absTol_ );

    flag = CVodeSetUserData( cvodeMem_, (void*) odeProblem_ );
	if( flag != CV_SUCCESS ) {
		std::ostringstream ss; ss << "Cvode set user data failed: " << flag;
		throw std::runtime_error(ss.str());
	}

    flag = CVodeSetInitStep( cvodeMem_, 0.0 );
	if( flag != CV_SUCCESS )
	   throw std::runtime_error( "CVodeSetInitStep failed" );

    flag = CVodeSetMaxNumSteps( cvodeMem_, maxNumSteps_ );
	if( flag != CV_SUCCESS )
	   throw std::runtime_error( "CVodeSetMaxNumSteps failed" );

    flag = CVodeSetMaxErrTestFails( cvodeMem_, 20 );
	if( flag != CV_SUCCESS )
	   throw std::runtime_error( "CVodeSetMaxErrTestFails failed" );

    flag = CVodeSetMaxConvFails( cvodeMem_, 50 );
	if( flag != CV_SUCCESS )
	   throw std::runtime_error( "CVodeSetMaxConvFails failed" );

	flag = CVDense( cvodeMem_, currentState_.size() );
	if( flag != CV_SUCCESS )
		throw std::runtime_error( "CVDense failed" );


	if( odeProblem_->hasJacobian() ) {
		flag = CVDlsSetDenseJacFn(cvodeMem_, OdeJacobian);
		if( flag != CV_SUCCESS )
			throw std::runtime_error( "CVDlsSetDenseJacFn failed" );
	}
}
#if 0
void CheckJacobian() {
	for(int i = 0; i < N; ++i) {
		const double eps = 1e-6;
		N_VScale_Serial(1.0, y, tmp1);
		NV_Ith_S(tmp1, i) += eps;
		OdeRhs(t, tmp1, tmp2, jac_data);
		N_VLinearSum_Serial(-1.0/eps, fy, 1.0/eps, tmp2, tmp1);
		N_VPrint_Serial(tmp1);
	}
}
#endif
#endif

