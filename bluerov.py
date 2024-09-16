from json import dump
from time import time

import matplotlib.pyplot as plt
from numpy import array, cross, diag, pi
from numpy.linalg import inv

from mpc import *
from utils import (
	build_inertial_matrix,
	build_transformation_matrix,
	check,
	generate_trajectory,
	gif_from_pngs,
	Logger,
	plot_bluerov,
	serialize_others,
	)


class Bluerov:
	state_size = 12
	actuation_size = 6

	mass = 11.5
	center_of_mass = array( [ 0.0, 0.0, 0.0 ] )
	weight = array( [ 0., 0., mass * 9.80665 ] )

	center_of_volume = array( [ 0.0, 0.0, -0.02 ] )
	buoyancy = array( [ 0, 0, -120. ] )

	inverse_inertial_matrix = inv(
			build_inertial_matrix( mass, center_of_mass, [ .16, .16, .16, 0., 0., 0. ] )
			)

	hydrodynamic_matrix = diag( [ 4.03, 6.22, 5.18, 0.07, 0.07, 0.07 ] )

	def __call__( self, state: ndarray, actuation: ndarray ) -> ndarray:
		transform_matrix = build_transformation_matrix( *state[ 3:6 ] )

		hydrostatic_forces = zeros( 6 )
		hydrostatic_forces[ :3 ] = transform_matrix[ :3, :3 ].T @ (Bluerov.weight + Bluerov.buoyancy)
		hydrostatic_forces[ 3: ] = cross(
				Bluerov.center_of_mass, transform_matrix[ :3, :3 ].T @ Bluerov.weight
				) + cross(
				Bluerov.center_of_volume, transform_matrix[ :3, :3 ].T @ Bluerov.buoyancy
				)

		xdot = zeros( state.shape )
		xdot[ :6 ] = transform_matrix @ state[ 6: ]
		xdot[ 6: ] = Bluerov.inverse_inertial_matrix @ (
				Bluerov.hydrodynamic_matrix @ state[ 6: ] + hydrostatic_forces + actuation)

		return xdot


if __name__ == "__main__":

	number_of_frames = 400
	time_step = 0.025

	initial_state = zeros( (Bluerov.state_size,) )
	initial_actuation = zeros( (Bluerov.actuation_size,) )

	horizon = 25

	key_frames = [ (0.0, [ 0., 0., 0., 0., 0., 0. ]), (.2, [ 0., 0., 1., 0., 0., 0. ]),
								 (.4, [ 0., 0., 1., 0., 0., -pi ]), (.6, [ -1., -1., 1., 0., 0., -pi ]),
								 (.8, [ -1., -1., 0., 0., 0., -pi ]), (1.0, [ 0., 0., 0., 0., 0., 0. ]),
								 (2., [ 0., 0., 0., 0., 0., 0. ]) ]

	trajectory = generate_trajectory(
			key_frames, 2 * number_of_frames
			)

	pose_weight_matrix = eye( Bluerov.state_size // 2 )
	pose_weight_matrix[ :3, :3 ] *= 2.
	actuation_weight_matrix = eye( Bluerov.actuation_size )
	actuation_weight_matrix[ :3, :3 ] *= 0.01
	final_cost_weight = 1.

	bluerov_model = Model(
			Bluerov(), time_step, initial_state, initial_actuation, record = True
			)

	bluerov_mpc = MPC(
			bluerov_model,
			horizon,
			trajectory[ 1:horizon + 1 ],
			time_steps_per_actuation = 25,
			tolerance = 1e-3,
			pose_weight_matrix = pose_weight_matrix,
			actuation_derivative_weight_matrix = actuation_weight_matrix,
			final_weight = final_cost_weight,
			record = True
			)

	logger = Logger()

	folder = (f'./export/bluerov_{int( time() )}')

	check( folder )
	check( f'{folder}/plot' )
	check( f'{folder}/data' )

	with open( f'{folder}/config.json', 'w' ) as f:
		dump( bluerov_mpc.__dict__, f, default = serialize_others )

	for frame in range( number_of_frames ):

		logger.log( f"frame {frame + 1}/{number_of_frames}" )

		bluerov_mpc.target_trajectory = trajectory[ frame + 1:frame + bluerov_mpc.horizon + 1 ]

		bluerov_mpc.compute_actuation()
		bluerov_mpc.apply_result()
		bluerov_model.step()

		logger.log( f"ux={bluerov_model.actuation[ :3 ]}" )
		logger.log( f"ut={bluerov_model.actuation[ 3:6 ]}" )
		logger.log( f"{bluerov_mpc.times[-1]=:.6f}s" )

		with open( f'{folder}/data/{frame}.json', 'w' ) as f:
			dump( bluerov_mpc.__dict__, f, default = serialize_others )

		fig = plot_bluerov(
			bluerov_mpc,
			frame = frame,
			n_frames = number_of_frames,
			full_trajectory = trajectory
			)
		plt.savefig( f'{folder}/plot/{frame}.png' )
		plt.close( 'all' )
		del fig

		logger.lognl( '' )
		logger.save_at( folder )

	gif_from_pngs( f'{folder}/plot' )
