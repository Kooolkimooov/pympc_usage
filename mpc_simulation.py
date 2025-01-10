from json import dump, load
from os.path import join, split
from time import perf_counter, time
from warnings import simplefilter

from numpy import array, cos, diff, eye, inf, pi, set_printoptions
from scipy.optimize import NonlinearConstraint

from pympc.models.dynamics.chain_of_four_with_usv import *
from pympc.models.model import Model
from pympc.controllers.mpc import MPC
from pympc.models.dynamics.seafloor import seafloor_function_0, SeafloorFromFunction
from utils import check, generate_trajectory, get_computer_info, Logger, print_dict, serialize_others


if __name__ == "__main__":
	simplefilter( 'ignore', RuntimeWarning )
	set_printoptions( precision = 2, linewidth = 10000, suppress = True )

	ti = perf_counter()

	n_frames = 500
	tolerance = 1e-6
	max_number_of_iteration = 100
	time_step = 0.1
	save_rate = int( .5 / time_step ) if time_step <= .1 else 1
	count_before_save = 0

	#@formatter:off
	key_frames = [
			(0., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
			(.5, [ -5., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
			(1., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18),
			(2., [ 2., 0., 0., 0., 0., 0. ] + [ 0. ] * 18)
			]
	# @formatter:on

	horizon = 5
	time_steps_per_actuation = 5
	time_step_prediction_factor = 2
	assert time_step_prediction_factor * horizon < n_frames / 2, 'scaled horizon is larger than the buffer at the end of the target trajectory'

	final_cost_weight = 0.
	objective_weight = .01

	sf_lb = 0.2
	sf_ub = inf
	dp_lb = 0.2
	dp_ub = 2.8
	dr_lb = -inf
	dr_ub = 2.8

	#@formatter:off
	constraints_values_labels = [
			'c_01_distance_to_seafloor',
			'c_12_distance_to_seafloor',
			'c_23_distance_to_seafloor',
			'br_0_distance_to_seafloor',
			'br_1_distance_to_seafloor',
			'br_2_distance_to_seafloor',
			'br_3_distance_to_seafloor',
			'br_0_br_1_horizontal_distance',
			'br_1_br_2_horizontal_distance',
			'br_2_br_3_horizontal_distance',
			'br_0_br_1_distance',
			'br_1_br_2_distance',
			'br_2_br_3_distance'
			]
	constraints_reason_labels = [
			'seafloor',
			'seafloor',
			'seafloor',
			'seafloor',
			'seafloor',
			'seafloor',
			'seafloor',
			'cable_length',
			'cable_length',
			'cable_length',
			'cable_length',
			'cable_length',
			'cable_length'
			]
	# @formatter:on

	constraint_lb_base = [ sf_lb, sf_lb, sf_lb, sf_lb, sf_lb, sf_lb, sf_lb, dp_lb, dp_lb, dp_lb, dr_lb, dr_lb, dr_lb ]
	constraint_ub_base = [ sf_ub, sf_ub, sf_ub, sf_ub, sf_ub, sf_ub, sf_ub, dp_ub, dp_ub, dp_ub, dr_ub, dr_ub, dr_ub ]

	assert (len( constraint_lb_base ) == len( constraints_values_labels )) and (
			len( constraint_ub_base ) == len( constraints_reason_labels )), 'bad definition of constraints'

	lb = [ constraint_lb_base ] * horizon
	ub = [ constraint_ub_base ] * horizon

	seafloor = SeafloorFromFunction( seafloor_function_0 )

	dynamics = ChainOf4WithUSV(
			water_surface_depth = 0.,
			water_current = array( [ .5, .5, 0. ] ),
			seafloor = seafloor,
			cables_length = 3.,
			cables_linear_mass = 0.01,
			get_cable_parameter_method = 'precompute'
			)

	trajectory = generate_trajectory( key_frames, 2 * n_frames )
	trajectory[ :, 0, dynamics.br_0_z ] = 1.5 * cos(
			1.25 * (trajectory[ :, 0, dynamics.br_0_position ][ :, 0 ] - 2) + pi
			) + 2.5

	max_required_speed = (max( norm( diff( trajectory[ :, 0, :3 ], axis = 0 ), axis = 1 ) ) / time_step)

	if 'y' != input( f'{max_required_speed=}, continue ? (y/n) ' ):
		exit()

	initial_state = zeros( (dynamics.state_size,) )
	initial_actuation = zeros( (dynamics.actuation_size,) )
	pose_weight_matrix = eye( initial_state.shape[ 0 ] // 2 )
	actuation_weight_matrix = eye( initial_actuation.shape[ 0 ] )

	initial_state[ dynamics.br_0_position ][ 0 ] = 2.
	initial_state[ dynamics.br_0_position ][ 2 ] = 1.
	initial_state[ dynamics.br_1_position ][ 0 ] = 2.5
	initial_state[ dynamics.br_1_position ][ 2 ] = 1.
	initial_state[ dynamics.br_2_position ][ 0 ] = 3.
	initial_state[ dynamics.br_2_position ][ 2 ] = 1.
	initial_state[ dynamics.br_3_position ][ 0 ] = 3.5
	initial_state[ dynamics.br_3_orientation ][ 2 ] = pi / 2
	actuation_weight_matrix[ dynamics.br_0_linear_actuation, dynamics.br_0_linear_actuation ] *= 0.
	actuation_weight_matrix[ dynamics.br_0_angular_actuation, dynamics.br_0_angular_actuation ] *= 1.
	actuation_weight_matrix[ dynamics.br_1_linear_actuation, dynamics.br_1_linear_actuation ] *= 0.
	actuation_weight_matrix[ dynamics.br_1_angular_actuation, dynamics.br_1_angular_actuation ] *= 1.
	actuation_weight_matrix[ dynamics.br_2_linear_actuation, dynamics.br_2_linear_actuation ] *= 0.
	actuation_weight_matrix[ dynamics.br_2_angular_actuation, dynamics.br_2_angular_actuation ] *= 1.
	actuation_weight_matrix[ dynamics.br_3_linear_actuation, dynamics.br_3_linear_actuation ] *= 0.
	actuation_weight_matrix[ dynamics.br_3_angular_actuation, dynamics.br_3_angular_actuation ] *= 0.
	pose_weight_matrix[ dynamics.br_0_position, dynamics.br_0_position ] *= 10.
	pose_weight_matrix[ dynamics.br_0_orientation, dynamics.br_0_orientation ] *= 1.
	pose_weight_matrix[ dynamics.br_1_position, dynamics.br_1_position ] *= 0.
	pose_weight_matrix[ dynamics.br_1_orientation, dynamics.br_1_orientation ] *= 1.
	pose_weight_matrix[ dynamics.br_2_position, dynamics.br_2_position ] *= 0.
	pose_weight_matrix[ dynamics.br_2_orientation, dynamics.br_2_orientation ] *= 1.
	pose_weight_matrix[ dynamics.br_3_position, dynamics.br_3_position ] *= 0.
	pose_weight_matrix[ dynamics.br_3_orientation, dynamics.br_3_orientation ] *= 0.

	model = Model(
			dynamics = dynamics,
			time_step = time_step,
			initial_state = initial_state,
			initial_actuation = initial_actuation,
			record = True
			)

	mpc = MPC(
			model = model,
			horizon = horizon,
			target_trajectory = trajectory,
			optimize_on = 'actuation',
			objective_weight = objective_weight,
			time_step_prediction_factor = time_step_prediction_factor,
			tolerance = tolerance,
			max_number_of_iteration = max_number_of_iteration,
			time_steps_per_actuation = time_steps_per_actuation,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True,
			# verbose = True
			)

	# inject constraints and objective as member functions so that they may access self
	mpc.constraints_function = chain_of_4_constraints.__get__( mpc, MPC )
	mpc.objective = chain_of_4_objective.__get__( mpc, MPC )

	constraint = NonlinearConstraint(
			mpc.constraints_function, array( lb ).flatten(), array( ub ).flatten()
			)
	constraint.value_labels = constraints_values_labels
	constraint.labels = constraints_reason_labels
	mpc.constraints = (constraint,)

	previous_nfeval_record = [ 0 ]
	previous_H01_record = [ 0. ]
	previous_H12_record = [ 0. ]
	previous_H23_record = [ 0. ]

	folder = join(
			split( __file__ )[ 0 ], 'export', split( __file__ )[ 1 ].split( '.' )[ 0 ] + '_' + str( int( time() ) )
			)

	if check( folder ) + check( f'{folder}/data' ):
		exit()

	logger = Logger()

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( mpc.__dict__ | get_computer_info() | { 'save_rate': save_rate }, f, default = serialize_others )

	with open( f'{folder}/config.json' ) as f:
		config = load( f )
		print_dict( config )

	if 'y' != input( 'run this simulation ? (y/n) ' ):
		exit()

	for frame in range( n_frames ):

		mpc.target_trajectory = trajectory[ frame + 1: ]

		logger.log( f'frame {frame + 1}/{n_frames} starts at t={perf_counter() - ti:.2f}' )

		model.actuation = mpc.compute_actuation()
		model.step()

		logger.log( f'ends at t={perf_counter() - ti:.2f}' )
		logger.log( f'{mpc.raw_result.message}' )
		logger.log( f'{mpc.raw_result.nit} iterations' )

		# try to recover if the optimization failed
		if not mpc.raw_result.success and mpc.tolerance < 1:
			mpc.tolerance *= 10
			logger.log( f'increasing tolerance: {mpc.tolerance:.0e}' )
		elif mpc.raw_result.success and mpc.tolerance > 2 * tolerance:
			# *2 because of floating point error
			mpc.tolerance /= 10
			logger.log( f'decreasing tolerance: {mpc.tolerance:.0e}' )
		else:
			logger.log( f'keeping tolerance: {mpc.tolerance:.0e}' )

		objective_value = mpc.get_objective()
		logger.log( f'objective: {objective_value:.2f}' )

		constraints_values = mpc.constraints_function( mpc.raw_result.x )
		logger.log( f'constraints: {constraints_values[ :len( constraint_lb_base ) ]}' )

		logger.lognl( '' )
		logger.save_at( folder )

		count_before_save += 1
		if count_before_save >= save_rate:
			count_before_save = 0
			print( 'saving state ...' )
			with open( f'{folder}/data/{int( frame / save_rate )}.json', 'w' ) as f:
				dump( mpc.__dict__, f, default = serialize_others )
