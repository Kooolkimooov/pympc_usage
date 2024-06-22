import json
import time
import glob
import traceback
from copy import deepcopy

import numpy as np

from bluerov import bluerov_configuration, robot
from mpc import Bounds, cost, optimize, NonlinearConstraint
from do_mpc_test import get_state_matrixes, do_mpc, casadi

test_param = {
		'state'             : np.array( [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ] ),
		'time_step'         : 0.5,
		'n_frames'          : 25,
		'robust_horizon'    : 4,
		'prediction_horizon': 10,
		'euclidean_cost'    : True,
		'final_cost'        : True,
		'target'            : np.array( [ .5, -.5, 1, 0, 0, np.pi ] )
		}

test_results = {
		'dts': [ ], 'costs': [ ], 'us': [ ], 'xs': [ ],
		}

test_bounds = {
		'state'             : [
				np.array( [ -1., -1., 0., -np.pi / 10, -np.pi / 10, -np.pi, 0., 0., 0., 0., 0., 0. ] ),
				np.array( [ 1., 1., 2., np.pi / 10, np.pi / 10, np.pi, 0., 0., 0., 0., 0., 0. ] ) ],
		'time_step'         : [ 0.001, 1 ],
		'n_frames'          : [ 25, 100 ],
		'robust_horizon'    : [ 1, 50 ],
		'prediction_horizon': [ 1, 100 ],
		'euclidean_cost'    : [ True, False ],
		'final_cost'        : [ True, False ],
		'target'            : [ np.array(
				[ -1., -1., 0., -np.pi / 10, -np.pi / 10, -np.pi / 10 ]
				), np.array( [ 1., 1., 2., np.pi / 10, np.pi / 10, np.pi / 10 ] ) ]
		}


def test_do_mpc( test_param: dict ) -> dict:

	horizon = test_param[ 'prediction_horizon' ]
	r_horizon = test_param[ 'robust_horizon' ]
	nsteps = test_param[ 'n_frames' ]
	dt = test_param[ 'time_step' ]
	target = casadi.SX( test_param[ 'target' ] )

	dts = [ ]
	costs = [ ]
	us = [ ]
	xs = [ ]

	model_type = 'continuous'  # either 'discrete' or 'continuous'
	model = do_mpc.model.Model( model_type )

	setup_mpc = {
			'n_horizon'  : horizon,
			'n_robust'   : r_horizon,
			'open_loop'  : False,
			't_step'     : dt,
			'nlpsol_opts': { 'ipopt.linear_solver': 'mumps' }
			}
	params_simulator = { 't_step': dt }

	eta = model.set_variable( '_x', 'eta', shape = (6, 1) )
	deta = model.set_variable( '_x', 'deta', shape = (6, 1) )
	nu = model.set_variable( '_x', 'nu', shape = (6, 1) )
	dnu = model.set_variable( '_x', 'dnu', shape = (6, 1) )

	u = model.set_variable( '_u', 'force', shape = (6, 1) )

	model.set_expression(
			'model_predictive_control_cost_function', casadi.sum1( (target - eta) ** 2 )
			)

	J, Iinv, D, S = get_state_matrixes( eta, bluerov_configuration )

	model.set_rhs( 'deta', J @ nu )
	model.set_rhs( 'dnu', Iinv @ (D @ nu + S + u) )
	model.set_rhs( 'eta', deta )
	model.set_rhs( 'nu', dnu )

	model.setup()

	mpc = do_mpc.controller.MPC( model )
	mpc.set_param( **setup_mpc )
	mpc.settings.supress_ipopt_output()

	lterm = casadi.SX( np.zeros( (1, 1) ) )
	mterm = casadi.SX( np.zeros( (1, 1) ) )
	if test_param[ 'euclidean_cost' ]:
		lterm = model.aux[ 'model_predictive_control_cost_function' ]
	if test_param[ 'final_cost' ]:
		mterm = model.aux[ 'model_predictive_control_cost_function' ]

	mpc.set_objective( mterm = mterm, lterm = lterm )
	mpc.set_rterm( force = 0 )

	# bounds on force
	mpc.bounds[ 'lower', '_u', 'force' ] = - bluerov_configuration[ "robot_max_actuation" ]
	mpc.bounds[ 'upper', '_u', 'force' ] = bluerov_configuration[ "robot_max_actuation" ]

	mpc.prepare_nlp()
	# bounds on force derivative
	for i in range( horizon - 1 ):
		for j in range( 6 ):
			mpc.nlp_cons.append( (mpc.opt_x[ '_u', i + 1, 0 ][ j ] - mpc.opt_x[ '_u', i, 0 ][ j ]) )
			mpc.nlp_cons_lb.append( - bluerov_configuration[ 'robot_max_actuation_ramp' ][ j ] * dt )
			mpc.nlp_cons_ub.append( bluerov_configuration[ 'robot_max_actuation_ramp' ][ j ] * dt )
	mpc.setup()

	simulator = do_mpc.simulator.Simulator( model )
	simulator.set_param( **params_simulator )
	simulator.setup()

	simulator.x0[ 'eta' ] = test_param[ 'state' ][ :6 ]
	simulator.x0[ 'nu' ] = np.zeros( 6 )

	x0 = simulator.x0
	mpc.x0 = x0
	mpc.u0 = np.zeros( 6 )
	mpc.set_initial_guess()

	for frame in range( nsteps ):
		print(
				f'{"do_mpc":<20}: frame {frame + 1:3} out of {test_param[ "n_frames" ]:<5}',
				end = '',
				flush = True
				)
		ti = time.perf_counter()
		u0 = mpc.make_step( x0 )
		tf = time.perf_counter()
		x0 = simulator.make_step( u0 )

		dts.append( tf - ti )
		costs.append( mpc.data[ '_aux' ][ 0 ][ -1 ] )
		xs.append( x0.flatten().tolist() )
		us.append( u0.flatten().tolist() )

		print(
				f'model_predictive_control_cost_function: {costs[ -1 ]:.3e} dt: {dts[ -1 ]:.3e} '
				f'pose: {x0[ 0 ][ 0 ]:+.3e}'
				f' {x0[ 1 ][ 0 ]:+.3e}'
				f' {x0[ 2 ][ 0 ]:+.3e}'
				f' {x0[ 3 ][ 0 ]:+.3e}'
				f' {x0[ 4 ][ 0 ]:+.3e}'
				f' {x0[ 5 ][ 0 ]:+.3e}', end = '\r', flush = True
				)

	print(
			f'{"do_mpc":<20}: done with final error '
			f'{np.abs( test_param[ "target" ] - test_param[ "state" ][ :6 ] )} '
			f'in {sum( dts )}s               '.replace(
					'\n', ''
					), flush = True
			)
	print(
			f'some stats on dts   : {np.percentile( dts, [ 1, 25, 50, 75, 99 ] )}'.replace( '\n', '' ),
			flush = True
			)
	print(
			f'some stats on costs : {np.percentile( costs, [ 1, 25, 50, 75, 99 ] )}'.replace( '\n', '' ),
			flush = True
			)

	do_mpc_test_results = { }

	do_mpc_test_results[ 'dts' ] = dts
	do_mpc_test_results[ 'costs' ] = costs
	do_mpc_test_results[ 'us' ] = us
	do_mpc_test_results[ 'xs' ] = xs

	return do_mpc_test_results


def test_inhouse_mpc( test_param: dict ) -> dict:

	model_args = { 'robot_configuration': bluerov_configuration }
	command_upper_bound = np.array(
			[ bluerov_configuration[ 'robot_max_actuation' ] ] * test_param[ 'robust_horizon' ]
			).flatten()
	command_lower_bound = - np.array(
			[ bluerov_configuration[ 'robot_max_actuation' ] ] * test_param[ 'robust_horizon' ]
			).flatten()
	command_derivative_upper_bound = np.array(
			[ bluerov_configuration[ 'robot_max_actuation_ramp' ] ] * test_param[ 'robust_horizon' ]
			).flatten()
	command_derivative_lower_bound = - np.array(
			[ bluerov_configuration[ 'robot_max_actuation_ramp' ] ] * test_param[ 'robust_horizon' ]
			).flatten()
	result_shape = (test_param[ 'robust_horizon' ], 6)
	result = np.zeros( result_shape )
	actuation = np.zeros( (6,) )

	dts = [ ]
	costs = [ ]
	dus = [ ]
	xs = [ ]

	state = deepcopy( test_param[ 'state' ] )

	for frame in range( test_param[ 'n_frames' ] ):
		print(
				f'{"inhouse":<20}: frame {frame + 1:3} out of {test_param[ "n_frames" ]:<5}',
				end = '',
				flush = True
				)

		ti = time.perf_counter()
		result = optimize(
				model = robot,
				cost_function = cost,
				target = test_param[ 'target' ],
				initial_guess = result,
				current_actuation = actuation,
				optimization_horizon = test_param[ 'robust_horizon' ],
				prediction_horizon = test_param[ 'prediction_horizon' ],
				state = state,
				time_step = test_param[ 'time_step' ],
				model_args = model_args,
				bounds = Bounds(
						command_derivative_lower_bound, command_derivative_upper_bound
						),
				constraints = (NonlinearConstraint(
						lambda u: (actuation + np.cumsum( u.reshape( result_shape ), axis = 0 )).flatten(),
						command_lower_bound,
						command_upper_bound
						),),
				activate_euclidean_cost = test_param[ 'euclidean_cost' ],
				activate_final_cost = test_param[ 'final_cost' ]
				)
		tf = time.perf_counter()

		result = result.reshape( result_shape )
		actuation += result[ 0 ]
		state += robot( state, actuation, **model_args ) * test_param[ 'time_step' ]

		dts.append( tf - ti )
		costs.append(
				cost(
						result,
						actuation,
						result_shape,
						robot,
						model_args,
						test_param[ 'target' ],
						test_param[ 'robust_horizon' ],
						test_param[ 'prediction_horizon' ],
						state,
						test_param[ 'time_step' ],
						None,
						test_param[ 'euclidean_cost' ],
						test_param[ 'final_cost' ],
						None,
						None,
						False
						)
				)

		dus.append( actuation.tolist() )
		xs.append( state.tolist() )

		print(
				f'model_predictive_control_cost_function: {costs[ -1 ]:.3e} dt: {dts[ -1 ]:.3e} '
				f'pose: {state[ 0 ]:+.3e}'
				f' {state[ 1 ]:+.3e}'
				f' {state[ 2 ]:+.3e}'
				f' {state[ 3 ]:+.3e}'
				f' {state[ 4 ]:+.3e}'
				f' {state[ 5 ]:+.3e}', end = '\r', flush = True
				)

	print(
			f'{"inhouse":<20}: done with final error '
			f'{np.abs( test_param[ "target" ] - state[ :6 ] )} '
			f'in {sum( dts )}s                  '.replace(
					'\n', ''
					), flush = True
			)
	print(
			f'some stats on dts   : {np.percentile( dts, [ 1, 25, 50, 75, 99 ] )}'.replace( '\n', '' ),
			flush = True
			)
	print(
			f'some stats on costs : {np.percentile( costs, [ 1, 25, 50, 75, 99 ] )}'.replace( '\n', '' ),
			flush = True
			)

	inhouse_test_results = { }

	inhouse_test_results[ 'dts' ] = dts
	inhouse_test_results[ 'costs' ] = costs
	inhouse_test_results[ 'us' ] = dus
	inhouse_test_results[ 'xs' ] = xs

	return inhouse_test_results


def get_random_test_param( out: dict, bounds: dict ):
	for key, val in bounds.items():
		u, l = val
		if isinstance( u, np.ndarray ):
			v = np.multiply( np.random.random( u.shape ), u - l ) + l
		elif isinstance( u, bool ):
			v = np.random.random() > .5
		elif isinstance( u, float ):
			v = (np.random.random()) * (u - l) + l
			if key == 'time_step':
				m = np.random.random() * 10
				e = np.random.random() * 3 + 1
				v = m * 10 ** (-e)
		elif isinstance( u, int ):
			v = int( (np.random.random()) * (u - l) + l )
		else:
			raise RuntimeError()
		out[ key ] = v

	if not out[ 'euclidean_cost' ] and not out[ 'final_cost' ]:
		if np.random.random() > .5:
			out[ 'euclidean_cost' ] = True
		else:
			out[ 'final_cost' ] = True


if __name__ == '__main__':

	save_file = 'test_suite.json'
	ntests = 100
	for i in range( ntests ):
		print( f'test {i + 1} out of {ntests}' )

		get_random_test_param( test_param, test_bounds )

		for key, value in test_param.items():
			print( f'{key:<20}: {value}'.replace( '\n', '' ) )
		print( '------------------------------------------------------ ' )

		# test inhouse mpc
		try:
			inhouse_test_results = test_inhouse_mpc( test_param )
			if inhouse_test_results.keys() != test_results.keys():
				raise ValueError( 'test results keys do not match' )
		except Exception as e:
			inhouse_test_results = f'dnf: {e}'
			print( f'\ninhouse failed: {e}' )  # print( traceback.format_exc() )
		except:
			inhouse_test_results = f'dnf: unknown'
			print( f'\ninhouse failed: unknown' )  # print( traceback.format_exc() )

		try:
			do_mpc_test_results = test_do_mpc( test_param )
		except Exception as e:
			do_mpc_test_results = f'dnf: {e}'
			print( f'\ndo_mpc failed: {e}' )  # print( traceback.format_exc() )
		except:
			do_mpc_test_results = f'dnf: unknown'
			print( f'\ndo_mpc failed: unknown' )  # print( traceback.format_exc() )

		test = {
				'test_param': test_param, 'inhouse_mpc': inhouse_test_results, 'do_mpc':
					do_mpc_test_results
				}

		test[ 'test_param' ][ 'state' ] = test[ 'test_param' ][ 'state' ].tolist()
		test[ 'test_param' ][ 'target' ] = test[ 'test_param' ][ 'target' ].tolist()

		if len( glob.glob( save_file ) ) > 0:
			with open( save_file ) as file:
				tests = json.load( file )
		else:
			tests = [ ]
		tests.append( test )
		with open( save_file, 'w' ) as file:
			json.dump( tests, file )
