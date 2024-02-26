import importlib.util
import numpy as np
import sys
from casadi import *

# Import do_mpc package:
import do_mpc

model_type = 'continuous'  # either 'discrete' or 'continuous'
model = do_mpc.model.Model( model_type )

bluerov_configuration = {
		"mass"                     : 11.5,
		"center_of_mass"           : np.array( [ 0.0, 0.0, 0.0 ] ),
		"buoyancy"                 : 120.0,
		"center_of_volume"         : np.array( [ 0.0, 0.0, - 0.02 ] ),
		"inertial_coefficients"    : np.array( [ .16, .16, .16, 0.0, 0.0, 0.0 ] ),
		"hydrodynamic_coefficients": np.array(
				[ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
				),
		"robot_max_actuation"      : np.array( [ 100, 100, 100, 100, 100, 100 ] ),
		"robot_max_actuation_ramp" : np.array( [ 100, 100, 100, 100, 100, 100 ] )
		}

def get_state_matrixes( state, robot_configuration = None ):
	if robot_configuration is None:
		robot_configuration = bluerov_configuration
	mass = robot_configuration[ "mass" ]
	inertial_coefficients = robot_configuration[ "inertial_coefficients" ]
	center_of_mass = robot_configuration[ "center_of_mass" ]
	I = np.eye( 6 )
	for i in range( 3 ):
		I[ i, i ] = mass
		I[ i + 3, i + 3 ] = inertial_coefficients[ i ]
	I[ 0, 4 ] = mass * center_of_mass[ 2 ]
	I[ 0, 5 ] = - mass * center_of_mass[ 1 ]
	I[ 1, 3 ] = - mass * center_of_mass[ 2 ]
	I[ 1, 5 ] = mass * center_of_mass[ 0 ]
	I[ 2, 3 ] = mass * center_of_mass[ 1 ]
	I[ 2, 4 ] = - mass * center_of_mass[ 0 ]
	I[ 4, 0 ] = mass * center_of_mass[ 2 ]
	I[ 5, 0 ] = - mass * center_of_mass[ 1 ]
	I[ 3, 1 ] = - mass * center_of_mass[ 2 ]
	I[ 5, 1 ] = mass * center_of_mass[ 0 ]
	I[ 3, 2 ] = mass * center_of_mass[ 1 ]
	I[ 4, 2 ] = - mass * center_of_mass[ 0 ]
	I[ 3, 4 ] = - inertial_coefficients[ 3 ]
	I[ 3, 5 ] = - inertial_coefficients[ 4 ]
	I[ 4, 5 ] = - inertial_coefficients[ 5 ]
	I[ 4, 3 ] = - inertial_coefficients[ 3 ]
	I[ 5, 3 ] = - inertial_coefficients[ 4 ]
	I[ 5, 4 ] = - inertial_coefficients[ 5 ]

	Phi, Theta, Psi = state[ 3 ], state[ 4 ], state[ 5 ]
	cPhi, sPhi = np.cos( Phi ), np.sin( Phi )
	cTheta, sTheta, tTheta = np.cos( Theta ), np.sin( Theta ), np.tan( Theta )
	cPsi, sPsi = np.cos( Psi ), np.sin( Psi )
	J = casadi.SX.zeros( (6, 6) )
	J[ 0, :3 ] = np.array(
			[ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi,
				sPsi * sPhi + cPsi * sTheta * cPhi ]
			)
	J[ 1, :3 ] = np.array(
			[ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi,
				-cPsi * sPhi + sPsi * sTheta * cPhi ]
			)
	J[ 2, :3 ] = np.array( [ -sTheta, cTheta * sPhi, cTheta * cPhi ] )
	J[ 3, 3: ] = np.array( [ 1, sPhi * tTheta, cPhi * tTheta ] )
	J[ 4, 3: ] = np.array( [ 0, cPhi, -sPhi ] )
	J[ 5, 3: ] = np.array( [ 0, sPhi / cTheta, cPhi / cTheta ] )

	hydrodynamic_coefficients = robot_configuration[ "hydrodynamic_coefficients" ]
	D = np.multiply( np.eye( 6 ), hydrodynamic_coefficients[ :6 ] ) + np.multiply(
			np.eye( 6 ), hydrodynamic_coefficients[ 6: ] * norm_2( state[ 6: ] )
			)
	buoyancy = robot_configuration[ "buoyancy" ]
	center_of_volume = robot_configuration[ "center_of_volume" ]
	Fw = mass * np.array( [ 0, 0, 9.80665 ] )
	Fb = buoyancy * np.array( [ 0, 0, 1 ] )
	S = casadi.SX( np.zeros( 6 ) )
	S[ :3 ] = J[ :3, :3 ].T @ (Fw + Fb)
	S[ 3: ] = J[ :3, :3 ].T @ (np.cross( center_of_mass, Fw ) + np.cross( center_of_volume, Fb ))

	return J, I, D, S

dt = 0.01
horizon = 100

eta = model.set_variable( '_x', 'eta', shape = (6, 1) )
deta = model.set_variable( '_x', 'deta', shape = (6, 1) )
nu = model.set_variable( '_x', 'nu', shape = (6, 1) )
dnu = model.set_variable( '_x', 'dnu', shape = (6, 1) )

u = model.set_variable( '_u', 'force', shape = (6, 1) )

cost = model.set_expression('cost', sum1((casadi.SX(np.array([0, 0, 1, 0, 0, 0])) - eta) ** 2))

J, I, D, S = get_state_matrixes(eta)

model.set_rhs( 'deta', J @ nu )
model.set_rhs( 'dnu', casadi.inv(I) @ (D @ eta + S + u) )
model.set_rhs('eta', eta + deta * dt)
model.set_rhs('nu', nu + dnu * dt)

model.setup()

mpc = do_mpc.controller.MPC( model )

setup_mpc = {
		'n_horizon'           : horizon,
		'n_robust'            : horizon // 4,
		'open_loop'           : 0,
		't_step'              : dt,
		'state_discretization': 'collocation',
		'collocation_type'    : 'radau',
		'collocation_deg'     : 3,
		'collocation_ni'      : 1,
		'store_full_solution' : True,  # Use MA27 linear solver in ipopt for faster calculations:
		'nlpsol_opts'         : { 'ipopt.linear_solver': 'mumps' }
		}
mpc.set_param( **setup_mpc )
mpc.settings.supress_ipopt_output()

mterm = model.aux[ 'cost' ]
lterm = model.aux[ 'cost' ]

mpc.set_objective( mterm = mterm, lterm = lterm )
# Input force is implicitly restricted through the objective.
mpc.set_rterm( force = 1e-9 )

# bounds on force
mpc.bounds[ 'lower', '_u', 'force' ] = - bluerov_configuration["robot_max_actuation"]
mpc.bounds[ 'upper', '_u', 'force' ] = bluerov_configuration["robot_max_actuation"]

mpc.prepare_nlp()

# bounds on force derivative
for i in range(horizon-1):
	for j in range(6):
		mpc.nlp_cons.append((mpc.opt_x['_u', i + 1, 0][j] - mpc.opt_x['_u', i, 0][j]) / dt)
		mpc.nlp_cons_lb.append(- bluerov_configuration['robot_max_actuation_ramp'][j])
		mpc.nlp_cons_ub.append(bluerov_configuration['robot_max_actuation_ramp'][j])

mpc.create_nlp()

mpc.setup()

estimator = do_mpc.estimator.StateFeedback( model )

simulator = do_mpc.simulator.Simulator( model )
params_simulator = {
		# Note: cvode doesn't support DAE systems.
		'integration_tool': 'idas', 'abstol': 1e-8, 'reltol': 1e-8, 't_step': dt
		}
simulator.set_param( **params_simulator )

simulator.setup()

simulator.x0['eta'] = np.zeros(6)
simulator.x0['nu'] = np.zeros(6)

x0 = simulator.x0
mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

nsteps = 100
for i in range( 1, nsteps + 1 ):
	print( f'{i = } / {nsteps = }' )
	u0 = mpc.make_step( x0 )
	y0 = estimator.make_step( u0 )
	x0 = simulator.make_step( y0 )
	print( f'{x0[:6].T = }' )
	print( f'{x0[6:12].T = }' )
	print( f'{x0[12:18].T = }' )
	print( f'{x0[18:].T = }' )
	print( f'{y0.T = }' )
	print( f'{u0.T = }' )
