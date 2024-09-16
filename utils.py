from glob import glob
from inspect import isfunction
from os import mkdir, path, remove

from cycler import cycler
from matplotlib import pyplot as plt
from numpy import array, cos, eye, meshgrid, nan, ndarray, ones, pi, sin, tan, zeros
from PIL import Image
from scipy.spatial.transform import Rotation

from calc_catenary_from_ext_points import get_coor_marker_points_ideal_catenary
from mpc import MPC


class Logger:
	def __init__( self, print_to_terminal: bool = True ):
		self.logs: str = ''
		self.print_to_terminal = print_to_terminal

	def log( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a tabulation
		:return: None
		'''
		self.logs += log
		self.logs += '\t'
		if self.print_to_terminal:
			print( log, end = '\t' )

	def lognl( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a new line
		:return: None
		'''
		self.logs += log
		self.logs += '\n'
		if self.print_to_terminal:
			print( log )

	def logrl( self, log: str ):
		'''
		:param log: text to be printed and saved. ends with a return to the beginning of the line,
		the saved text goes to a new line
		:return: None
		'''
		self.logs += log
		self.logs += '\n'
		if self.print_to_terminal:
			print( log, end = '\r' )

	def save_at( self, path: str, file: str = 'logs' ):
		"""
		:param path: folder in which to save the current log
		:param file: name of the file
		"""
		with open( f'{path}/logs.txt', 'w' ) as f:
			f.write( self.logs )


def generate_trajectory(
		key_frames: list[ tuple[ float, list ] ], n_points: int
		):
	assert key_frames[ 0 ][ 0 ] == 0., "trajectory doesn't start at t = 0."

	n_dim = len( key_frames[ 0 ][ 1 ] )
	timespan = key_frames[ -1 ][ 0 ]
	trajectory = zeros( (n_points, 1, n_dim) )
	start_point = 0

	for frame_index in range( len( key_frames ) - 1 ):
		frame_0 = key_frames[ frame_index ]
		frame_1 = key_frames[ frame_index + 1 ]
		sub_timespan = frame_1[ 0 ] - frame_0[ 0 ]
		sub_n_points = int( n_points * sub_timespan / timespan )

		funcs = [ ]
		for dim in range( n_dim ):
			funcs += [ cubic_interpolation_function( frame_0[ 1 ][ dim ], frame_1[ 1 ][ dim ], 0., 0. ) ]

		for point in range( sub_n_points ):
			for dim in range( n_dim ):
				trajectory[ start_point + point, :, dim ] = funcs[ dim ]( point / sub_n_points )

		start_point += sub_n_points
	for dim in range( n_dim ):
		trajectory[ -1, :, dim ] = key_frames[ -1 ][ 1 ][ dim ]
	return trajectory


def cubic_interpolation_function( f_0: float, f_1: float, f_0p: float, f_1p: float ):
	a = 2 * f_0 - 2 * f_1 + f_0p + f_1p
	b = -3 * f_0 + 3 * f_1 - 2 * f_0p - f_1p
	c = f_0p
	d = f_0

	def function( x: float ) -> float:
		return a * pow( x, 3 ) + b * pow( x, 2 ) + c * x + d

	return function


def check( folder: str ):
	if path.exists( folder ):
		files_in_dir = glob( f'{folder}/*' )
		if len( files_in_dir ) > 0:
			if input( f"{folder} contains data. Remove? (y/n) " ) == 'y':
				for fig in files_in_dir:
					remove( fig )
			else:
				exit()
	else:
		mkdir( folder )


def gif_from_pngs( folder: str ):
	names = [ image for image in glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/animation.gif", append_images = frames, loop = True, save_all = True
			)


def serialize_others( obj: any ):
	exclude = {'__doc__': None, '__init__': None, '__module__': None, '__weakref__': None}
	if isfunction( obj ):
		return obj.__name__
	if isinstance( obj, ndarray ):
		return obj.tolist()
	if isinstance(obj, slice):
		return f'{obj.start}:{obj.stop}:{obj.step}'
	if isinstance(obj, bool):
		return str(obj)
	try:
		return { k: v for k, v in (obj.__dict__ | obj.__class__.__dict__).items() if k not in exclude }
	except:
		return 'unable to process'


def print_dict( d: dict, prefix: str = '' ):
	for k, v in d.items():

		if isinstance( v, dict ):
			print( prefix + k + ':' )
			print_dict( v, prefix + '\t' )
			continue

		if isinstance( v, list ):
			if len( v ) > 0 and isinstance( v[ 0 ], dict ):
				print( prefix + k + ':' )
				print_dict( { str( i ): e for i, e in enumerate( v ) }, prefix + '\t' )
				continue

			l = array( v ).shape
			print( prefix + k + ':', v if sum( l ) < 10 else l )
			continue

		print( prefix + k + ':', v )


def build_transformation_matrix( phi: float, theta: float, psi: float ) -> ndarray:
	cPhi, sPhi = cos( phi ), sin( phi )
	cTheta, sTheta, tTheta = cos( theta ), sin( theta ), tan( theta )
	cPsi, sPsi = cos( psi ), sin( psi )

	matrix = zeros( (6, 6) )
	matrix[ 0, :3 ] = array(
			[ cPsi * cTheta, -sPsi * cPhi + cPsi * sTheta * sPhi, sPsi * sPhi + cPsi * sTheta * cPhi ]
			)
	matrix[ 1, :3 ] = array(
			[ sPsi * cTheta, cPsi * cPhi + sPsi * sTheta * sPhi, -cPsi * sPhi + sPsi * sTheta * cPhi ]
			)
	matrix[ 2, :3 ] = array( [ -sTheta, cTheta * sPhi, cTheta * cPhi ] )
	matrix[ 3, 3: ] = array( [ 1, sPhi * tTheta, cPhi * tTheta ] )
	matrix[ 4, 3: ] = array( [ 0, cPhi, -sPhi ] )
	matrix[ 5, 3: ] = array( [ 0, sPhi / cTheta, cPhi / cTheta ] )
	return matrix


def build_inertial_matrix(
		mass: float, center_of_mass: ndarray, inertial_coefficients: list[ float ]
		) -> ndarray:
	inertial_matrix = eye( 6 )
	for i in range( 3 ):
		inertial_matrix[ i, i ] = mass
		inertial_matrix[ i + 3, i + 3 ] = inertial_coefficients[ i ]
	inertial_matrix[ 0, 4 ] = mass * center_of_mass[ 2 ]
	inertial_matrix[ 0, 5 ] = - mass * center_of_mass[ 1 ]
	inertial_matrix[ 1, 3 ] = - mass * center_of_mass[ 2 ]
	inertial_matrix[ 1, 5 ] = mass * center_of_mass[ 0 ]
	inertial_matrix[ 2, 3 ] = mass * center_of_mass[ 1 ]
	inertial_matrix[ 2, 4 ] = - mass * center_of_mass[ 0 ]
	inertial_matrix[ 4, 0 ] = mass * center_of_mass[ 2 ]
	inertial_matrix[ 5, 0 ] = - mass * center_of_mass[ 1 ]
	inertial_matrix[ 3, 1 ] = - mass * center_of_mass[ 2 ]
	inertial_matrix[ 5, 1 ] = mass * center_of_mass[ 0 ]
	inertial_matrix[ 3, 2 ] = mass * center_of_mass[ 1 ]
	inertial_matrix[ 4, 2 ] = - mass * center_of_mass[ 0 ]
	inertial_matrix[ 3, 4 ] = - inertial_coefficients[ 3 ]
	inertial_matrix[ 3, 5 ] = - inertial_coefficients[ 4 ]
	inertial_matrix[ 4, 5 ] = - inertial_coefficients[ 5 ]
	inertial_matrix[ 4, 3 ] = - inertial_coefficients[ 3 ]
	inertial_matrix[ 5, 3 ] = - inertial_coefficients[ 4 ]
	inertial_matrix[ 5, 4 ] = - inertial_coefficients[ 5 ]

	return inertial_matrix


def plot_bluerov( mpc: MPC, **kwargs ):
	# we record the initial value + the new value after the integration in `step()`
	time_previous = [ i * mpc.model.time_step - (kwargs[ 'frame' ] + 1) * mpc.model.time_step for
										i in
										range( len( mpc.model.previous_states ) ) ]
	time_prediction = [ i * mpc.model.time_step for i in
											range( mpc.predicted_trajectories[ 0 ].shape[ 0 ] - 1 ) ]

	fig = plt.figure()
	view = plt.subplot2grid( (3, 5), (0, 0), 4, 3, fig, projection = '3d' )
	view.set_xlabel( "x" )
	view.set_ylabel( "y" )
	view.set_xlim( -2, 2 )
	view.set_ylim( -2, 2 )
	view.set_zlim( 0, 4 )
	view.invert_yaxis()
	view.invert_zaxis()

	ax_pos = plt.subplot2grid( (3, 5), (0, 3), 1, 2, fig )
	ax_pos.set_ylabel( 'position' )
	ax_pos.yaxis.set_label_position( "right" )
	ax_pos.yaxis.tick_right()
	ax_pos.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_pos.set_ylim( -3, 3 )
	ax_pos.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

	ax_ang = plt.subplot2grid( (3, 5), (1, 3), 1, 2, fig )
	ax_ang.set_ylabel( 'angle' )
	ax_ang.yaxis.set_label_position( "right" )
	ax_ang.yaxis.tick_right()
	ax_ang.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_ang.set_ylim( -2 * pi, 2 * pi )
	ax_ang.set_prop_cycle( cycler( 'color', [ 'blue', 'red', 'green' ] ) )

	ax_act = plt.subplot2grid( (3, 5), (2, 3), 1, 2, fig )
	ax_act.set_ylabel( 'actuation' )
	ax_act.set_xlabel( 'time' )
	ax_act.yaxis.set_label_position( "right" )
	ax_act.yaxis.tick_right()
	ax_act.set_xlim( time_previous[ 0 ], time_prediction[ -1 ] )
	ax_act.set_prop_cycle(
			cycler( 'color', [ 'blue', 'red', 'green', 'cyan', 'orange', 'olive' ] )
			)

	plt.subplots_adjust( hspace = 0., wspace = .5 )
	fig.suptitle( f"{kwargs[ 'frame' ] + 1}/{kwargs[ 'n_frames' ]} - {mpc.times[ -1 ]:.6f}s" )

	state_r = Rotation.from_euler( 'xyz', mpc.model.state[ 3:6 ] ).as_matrix()
	target_r = Rotation.from_euler( 'xyz', mpc.target_trajectory[ 0, 0, 3:6 ] ).as_matrix()

	quiver_scale = .5
	view.quiver( *mpc.model.state[ :3 ], *(state_r @ (quiver_scale * array( [ 1., 0., 0. ] ))) )
	view.quiver(
			*mpc.target_trajectory[ 0, 0, :3 ], *(target_r @ (quiver_scale * array( [ 1., 0., 0. ] )))
			)

	view.plot(
			array( mpc.model.previous_states )[ :, 0 ],
			array( mpc.model.previous_states )[ :, 1 ],
			array( mpc.model.previous_states )[ :, 2 ],
			'b'
			)

	ax_pos.plot(
			time_previous + time_prediction[ 1: ],
			kwargs[ 'full_trajectory' ][ :len( time_previous ) + len( time_prediction ) - 1, 0, 0 ],
			':b'
			)

	ax_pos.plot(
			time_previous + time_prediction[ 1: ],
			kwargs[ 'full_trajectory' ][ :len( time_previous ) + len( time_prediction ) - 1, 0, 1 ],
			':b'
			)

	ax_pos.plot(
			time_previous + time_prediction[ 1: ],
			kwargs[ 'full_trajectory' ][ :len( time_previous ) + len( time_prediction ) - 1, 0, 2 ],
			':b'
			)

	view.plot(
			kwargs[ 'full_trajectory' ][ :, 0, 0 ],
			kwargs[ 'full_trajectory' ][ :, 0, 1 ],
			kwargs[ 'full_trajectory' ][ :, 0, 2 ],
			':'
			)

	previous_pos_record_array = array( mpc.model.previous_states )[ :, :3 ]
	previous_ang_record_array = array( mpc.model.previous_states )[ :, 3:6 ]

	ax_pos.plot( time_previous, previous_pos_record_array )
	ax_ang.plot( time_previous, previous_ang_record_array )
	ax_act.plot( time_previous, mpc.model.previous_actuations )

	step = 1
	if len( mpc.predicted_trajectories ) > 1000:
		step = len( mpc.predicted_trajectories ) // 1000

	for f_eval in range( 0, len( mpc.predicted_trajectories ), step ):
		pos_record_array = mpc.predicted_trajectories[ f_eval ][ 1:, 0, :3 ]
		ang_record_array = mpc.predicted_trajectories[ f_eval ][ 1:, 0, 3:6 ]

		view.plot(
				pos_record_array[ :, 0 ],
				pos_record_array[ :, 1 ],
				pos_record_array[ :, 2 ],
				'b',
				linewidth = .1
				)

		ax_pos.plot( time_prediction, pos_record_array, linewidth = .1 )
		ax_ang.plot( time_prediction, ang_record_array, linewidth = .1 )
		ax_act.plot(
				time_prediction, mpc.candidate_actuations[ f_eval ][ 1:, 0, : ], linewidth = .1
				)

	# plot vertical line from y min to y max
	ax_pos.axvline( color = 'g' )
	ax_ang.axvline( color = 'g' )
	ax_act.axvline( color = 'g' )

	return fig
