from glob import glob
from inspect import isfunction, ismethod
from os import mkdir, path, remove

from numpy import array, cos, eye, ndarray, sin, tan, zeros
from PIL import Image

G: float = 9.80665

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


def check( folder: str, recursive = False ) -> int:
	'''
	counts the number of objects in a folder and removes them if the user agrees
	if the folder does not exist, it creates it
	:param folder:
	:param recursive:
	:return: number of objects in the folder
	'''
	n = 0
	if path.exists( folder ):
		objects_in_dir = glob( f'{folder}/*' )
		n += len( objects_in_dir )
		if n > 0:
			if input( f"{folder} exists and contains data. Remove? (y/n) " ) == 'y':
				for object in objects_in_dir:
					if path.isdir( object ) and recursive:
						n += check( object )
					else:
						print( f'removing {object}' )
						remove( object )
						n -= 1
	else:
		mkdir( folder )

	return n


def gif_from_pngs( folder: str, duration: float = None ):
	if duration is None:
		duration = 33.
	names = [ image for image in glob( f"{folder}/*.png" ) ]
	names.sort( key = lambda x: path.getmtime( x ) )
	frames = [ Image.open( name ) for name in names ]
	frame_one = frames[ 0 ]
	frame_one.save(
			f"{folder}/animation.gif", append_images = frames, loop = True, save_all = True, duration = duration
			)


def serialize_others( obj: any ):
	if isfunction( obj ) or ismethod( obj ):
		return obj.__name__
	if isinstance( obj, ndarray ):
		return obj.tolist()
	if isinstance( obj, slice ):
		return f'{obj.start}:{obj.stop}:{obj.step}'
	if isinstance( obj, bool ):
		return str( obj )
	try:
		output = { }
		# class attribute priority is for left most class in inheritance list,
		# we reverse the __bases__ list to get the correct order
		for base in reversed( get_all_bases( obj.__class__ ) ):
			output |= base.__dict__
		output |= obj.__class__.__dict__
		output |= obj.__dict__

		return output
	except:
		return 'unable to process'


def get_all_bases( obj: any ):
	bases = obj.__bases__
	for base in bases:
		if base.__name__ == 'object':
			continue
		bases += get_all_bases( base )
	return bases


def print_dict( d: dict, prefix: str = '' ):
	for k, v in d.items():

		if k[ 0 ] == '_':
			continue

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
