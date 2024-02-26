import casadi
import numpy as np
from casadi import *

bluerov_configuration = {
		"mass"                     : 11.5,
		"center_of_mass"           : np.array( [ 0.0, 0.0, 0.0 ] ),
		"buoyancy"                 : 120.0,
		"center_of_volume"         : np.array( [ 0.0, 0.0, - 0.02 ] ),
		"inertial_coefficients"    : np.array( [ .16, .16, .16, 0.0, 0.0, 0.0 ] ),
		"hydrodynamic_coefficients": np.array(
				[ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
				),
		"robot_max_actuation"      : np.array( [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ] ),
		"robot_max_actuation_ramp" : np.array( [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ] )
		}


def compute_IAb( state: SX, robot_configuration = None ) -> tuple:

	global bluerov_configuration
	if robot_configuration is None:
		robot_configuration = bluerov_configuration

	mass = robot_configuration[ "mass" ]
	inertial_coefficients = robot_configuration[ "inertial_coefficients" ]
	center_of_mass = robot_configuration[ "center_of_mass" ]
	I = np.eye( 12 )
	for i in range( 6, 9 ):
		I[ i, i ] = mass
		I[ i + 3, i + 3 ] = inertial_coefficients[ i - 6 ]
	I[ 6, 10 ] = mass * center_of_mass[ 2 ]
	I[ 6, 11 ] = - mass * center_of_mass[ 1 ]
	I[ 7, 9 ] = - mass * center_of_mass[ 2 ]
	I[ 7, 11 ] = mass * center_of_mass[ 0 ]
	I[ 8, 9 ] = mass * center_of_mass[ 1 ]
	I[ 8, 10 ] = - mass * center_of_mass[ 0 ]
	I[ 10, 6 ] = mass * center_of_mass[ 2 ]
	I[ 11, 6 ] = - mass * center_of_mass[ 1 ]
	I[ 9, 7 ] = - mass * center_of_mass[ 2 ]
	I[ 11, 7 ] = mass * center_of_mass[ 0 ]
	I[ 9, 8 ] = mass * center_of_mass[ 1 ]
	I[ 10, 8 ] = - mass * center_of_mass[ 0 ]
	I[ 9, 10 ] = - inertial_coefficients[ 3 ]
	I[ 9, 11 ] = - inertial_coefficients[ 4 ]
	I[ 10, 11 ] = - inertial_coefficients[ 5 ]
	I[ 10, 9 ] = - inertial_coefficients[ 3 ]
	I[ 11, 9 ] = - inertial_coefficients[ 4 ]
	I[ 11, 10 ] = - inertial_coefficients[ 5 ]

	Phi, Theta, Psi = state[ 3 ], state[ 4 ], state[ 5 ]
	cPhi, sPhi = np.cos( Phi ), np.sin( Phi )
	cTheta, sTheta, tTheta = np.cos( Theta ), np.sin( Theta ), np.tan( Theta )
	cPsi, sPsi = np.cos( Psi ), np.sin( Psi )

	J = np.zeros( (6, 6) )
	J[ 0, :3 ] = [ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi,
								 sPsi * sPhi + cPsi * sTheta * cPhi ]
	J[ 1, :3 ] = [ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi,
								 -cPsi * sPhi + sPsi * sTheta * cPhi ]
	J[ 2, :3 ] = [ -sTheta, cTheta * sPhi, cTheta * cPhi ]
	J[ 3, 3: ] = [ 1, sPhi * tTheta, cPhi * tTheta ]
	J[ 4, 3: ] = [ 0, cPhi, -sPhi ]
	J[ 5, 3: ] = [ 0, sPhi / cTheta, cPhi / cTheta ]
	hydrodynamic_coefficients = robot_configuration[ "hydrodynamic_coefficients" ]
	D = np.multiply( np.eye( 6 ), hydrodynamic_coefficients[ :6 ] ) + np.multiply(
			np.eye( 6 ), hydrodynamic_coefficients[ 6: ] * norm_2( state[ 6: ] )
			)

	A = casadi.SX( np.zeros( (12, 12) ) )
	A[ 6:, :6 ] = J
	A[ 6:, 6: ] = D

	buoyancy = robot_configuration[ "buoyancy" ]
	center_of_volume = robot_configuration[ "center_of_volume" ]
	Fw = mass * np.array( [ 0, 0, 9.80665 ] )
	Fb = buoyancy * np.array( [ 0, 0, 1 ] )
	b = casadi.SX( np.zeros( 12 ) )
	b[ 6:9 ] = J[ :3, :3 ].T @ (Fw + Fb)
	b[ 9: ] = J[ :3, :3 ].T @ (np.cross( center_of_mass, Fw ) + np.cross( center_of_volume, Fb ))

	return I, A, b


if __name__ == '__main__':
	import importlib.util
	import do_mpc

	model = do_mpc.model.Model( 'continuous' )

	state = model.set_variable( var_type = '_x', var_name = 'state', shape = (12, 1) )
	actuation = model.set_variable( var_type = '_u', var_name = 'actuation', shape = (6, 1) )
	# dactuation = model.set_variable( var_type = '_x', var_name = 'dactuation', shape = (6, 1) )

	dt = 0.01

	I, A, b = compute_IAb( state )
	u = casadi.SX( np.zeros( 12 ) )
	u[ 6: ] = actuation
	dstate = casadi.inv( I ) @ (A @ state + b + u)

	model.set_rhs( 'state', dstate )
	# model.set_rhs( 'dactuation', casadi.diff( actuation ) )

	model.set_expression( expr_name = 'cost', expr = sum1( state ** 2 ) )

	# print(model.x)
	# print(model.u)
	# print(model.rhs_list)

	model.setup()

	mpc = do_mpc.controller.MPC( model )

	setup_mpc = {
			'n_robust'            : 5,
			'n_horizon'           : 20,
			't_step'              : dt,
			'state_discretization': 'collocation',
			'collocation_type'    : 'radau',
			'collocation_deg'     : 3,
			'collocation_ni'      : 1,
			# 'nlpsol_opts'         : { 'ipopt.linear_solver': 'MA27' }
			}

	mpc.set_param( **setup_mpc )
	mpc.set_objective( mterm = model.aux[ 'cost' ], lterm = model.aux[ 'cost' ] )
	mpc.set_rterm( actuation = 1e-5 )

	mpc.bounds[ 'lower', '_u', 'actuation' ] = - bluerov_configuration[ "robot_max_actuation" ]
	mpc.bounds[ 'upper', '_u', 'actuation' ] = bluerov_configuration[ "robot_max_actuation" ]

	# mpc.bounds[ 'lower', '_x', 'dactuation' ] = -
	# bluerov_configuration[	# 	"robot_max_actuation_ramp" ] * dt	# mpc.bounds[ 'upper', '_x',
	# 'dactuation' ] = bluerov_configuration[	#
	# "robot_max_actuation_ramp" ] * dt

	mpc.setup()

	params_simulator = {
			'integration_tool': 'idas',
			'abstol'          : 1e-8,
			'reltol'          : 1e-8,
			't_step'          : 0.04
			}
	simulator = do_mpc.simulator.Simulator( model )
	simulator.set_param( **params_simulator )
	simulator.setup()

	x = np.array( [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] )
	mpc.x0 = x
	simulator.x0 = x

	mpc.set_initial_guess()

	for k in range( 100 ):
		u = mpc.make_step( x )
		x = simulator.make_step( u )

		print( x )
		print( u )
