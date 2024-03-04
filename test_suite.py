from bluerov import robot, bluerov_configuration
from mpc import model_predictive_control, cost, NonlinearConstraint, Bounds
import numpy as np
import time
import json

test_param = {
		'state'             : np.array( [ 0., 0, 0, 0, 0., 0., 0., 0, 0, 0, 0., 0. ] ),
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
				np.array( [ -1., -1., -1., -np.pi, -np.pi, -np.pi, 0., 0., 0., 0., 0., 0. ] ),
				np.array( [ 1., 1., 1., np.pi, np.pi, np.pi, 0., 0., 0., 0., 0., 0. ] ) ],

		'time_step'         : [ 0.001, 1 ],
		'n_frames'          : [ 25, 250 ],
		'robust_horizon'    : [ 1, 100 ],
		'prediction_horizon': [ 1, 100 ],
		'euclidean_cost'    : [ True, False ],
		'final_cost'        : [ True, False ],
		'target'            : [ np.array(
				[ -1., -1., -1., -np.pi, -np.pi, -np.pi ]
				), np.array( [ 1., 1., 1., np.pi, np.pi, np.pi ] ) ]
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
		ti = time.perf_counter()
		result = model_predictive_control(
				model = robot,
				cost = cost,
				target = test_param[ 'target' ],
				last_result = result,
				current_actuation = actuation,
				robust_horizon = test_param[ 'robust_horizon' ],
				prediction_horizon = test_param[ 'prediction_horizon' ],
				state = test_param[ 'state' ],
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
		test_param[ 'state' ] += robot( test_param[ 'state' ], actuation, **model_args ) * test_param[
			'time_step' ]

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
						test_param[ 'state' ],
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

	test_results[ 'dts' ] = dts
	test_results[ 'costs' ] = costs
	test_results[ 'dus' ] = dus
	test_results[ 'xs' ] = xs
	return test_results


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
		elif isinstance( u, int ):
			v = int( (np.random.random()) * (u - l) + l )
		else:
			raise RuntimeError()
		out[ key ] = v


if __name__ == '__main__':

	tests = [ ]
	for _ in range( 1 ):

		get_random_test_param( test_param, test_bounds )
		test = {
				'test_param': test_param, 'inhouse_mpc': None, 'do_mpc': None
				}
		for key, value in test_param.items():
			print( f'{key}: {value}'.replace( '\n', '' ) )

		# test inhouse mpc
		try:
			test_results = test_inhouse_mpc( test_param )
			test['inhouse_mpc'] = test_results
		except:
			test['inhouse_mpc'] = 'dnf'


		tests.append( test )
		with open( 'test_suite.json', 'w' ) as file:
			json.dump( tests, file )
