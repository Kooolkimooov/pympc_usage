import json
import time
import traceback
from copy import deepcopy

import numpy as np

from bluerov import bluerov_configuration, robot
from mpc import Bounds, cost, model_predictive_control, NonlinearConstraint

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
		'dts': [ ], 'costs': [ ], 'dus': [ ], 'xs': [ ],
		}

test_bounds = {
		'state'             : [
				np.array( [ -1., -1., 0., -np.pi / 10, -np.pi / 10, -np.pi, 0., 0., 0., 0., 0., 0. ] ),
				np.array( [ 1., 1., 2., np.pi / 10, np.pi / 10, np.pi, 0., 0., 0., 0., 0., 0. ] ) ],

		'time_step'         : [ 0.001, 1 ],
		'n_frames'          : [ 25, 250 ],
		'robust_horizon'    : [ 1, 100 ],
		'prediction_horizon': [ 1, 100 ],
		'euclidean_cost'    : [ True, False ],
		'final_cost'        : [ True, False ],
		'target'            : [ np.array(
				[ -1., -1., 0., -np.pi / 10, -np.pi / 10, -np.pi / 10 ]
				), np.array( [ 1., 1., 2., np.pi / 10, np.pi / 10, np.pi / 10 ] ) ]
		}


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

	for frame in range( test_param[ 'n_frames' ] ):
		print(
				f'inhouse: frame {frame + 1:3} out of {test_param[ "n_frames" ]:<5}', end = '',
				flush = True
				)
		state = deepcopy( test_param[ 'state' ] )

		ti = time.perf_counter()
		result = model_predictive_control(
				model = robot,
				cost = cost,
				target = test_param[ 'target' ],
				last_result = result,
				current_actuation = actuation,
				robust_horizon = test_param[ 'robust_horizon' ],
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
		test_param[ 'state' ] += robot( state, actuation, **model_args ) * test_param[ 'time_step' ]

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

		dus.append( result.tolist() )
		xs.append( test_param[ 'state' ].tolist() )

		print(
				f'cost: {costs[ -1 ]:.3e} dt: {dts[ -1 ]:.3e} '
				f'pose: {test_param[ "state" ][ 0 ]:+.3e}'
				f' {test_param[ "state" ][ 1 ]:+.3e}'
				f' {test_param[ "state" ][ 2 ]:+.3e}'
				f' {test_param[ "state" ][ 3 ]:+.3e}'
				f' {test_param[ "state" ][ 4 ]:+.3e}'
				f' {test_param[ "state" ][ 5 ]:+.3e}', end = '\r', flush = True
				)

	print(
			f'\ninhouse done with final error '
			f'{test_param[ 'target' ] - test_param[ 'state' ][ :6 ]}'.replace(
					'\n', ''
					), flush = True
			)
	print( f'some stats on dts   : {np.percentile( dts, [ 1, 25, 50, 75, 99 ] )}', flush = True )
	print( f'some stats on costs : {np.percentile( costs, [ 1, 25, 50, 75, 99 ] )}', flush = True )

	inhouse_test_results = { }

	inhouse_test_results[ 'dts' ] = dts
	inhouse_test_results[ 'costs' ] = costs
	inhouse_test_results[ 'dus' ] = dus
	inhouse_test_results[ 'xs' ] = xs

	return inhouse_test_results


def get_random_test_param( out: dict, bounds: dict ):
	for key, val in bounds.items():
		u, l = val
		v = None
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

	ntests = 10
	for i in range( ntests ):
		print( f'test {i + 1} out of {ntests}' )

		get_random_test_param( test_param, test_bounds )

		for key, value in test_param.items():
			print( f'{key:<20}: {value}'.replace( '\n', '' ) )

		# test inhouse mpc
		try:
			inhouse_test_results = test_inhouse_mpc( test_param )
			if inhouse_test_results.keys() != test_results.keys():
				raise ValueError( 'test results keys do not match' )
		except Exception as e:
			inhouse_test_results = f'dnf: {e}'
			print( f'\ninhouse failed: {e}' )
			print( traceback.format_exc() )
			if 'str' in str(e):
				exit(-1)

		try:
			do_mpc_test_results = test_do_mpc( test_param )
		except Exception as e:
			do_mpc_test_results = f'dnf: {e}'
			# print( f'\ndo_mpc failed: {e}' )
			# print( traceback.format_exc() )

		test = {
				'test_param': test_param, 'inhouse_mpc': inhouse_test_results, 'do_mpc':
					do_mpc_test_results
				}

		test[ 'test_param' ][ 'state' ] = test[ 'test_param' ][ 'state' ].tolist()
		test[ 'test_param' ][ 'target' ] = test[ 'test_param' ][ 'target' ].tolist()

		with open( 'test_suite.json' ) as file:
			tests = json.load( file )
		tests.append( test )
		with open( 'test_suite.json', 'w' ) as file:
			json.dump( tests, file )
