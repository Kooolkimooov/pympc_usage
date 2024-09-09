from numpy import ndarray, zeros
from scipy.optimize import NonlinearConstraint


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


def serialize_others( obj: any ) -> str:
	if callable( obj ):
		return obj.__name__
	if isinstance( obj, ndarray ):
		return obj.tolist()
	if isinstance( obj, NonlinearConstraint ):
		return obj.__dict__
