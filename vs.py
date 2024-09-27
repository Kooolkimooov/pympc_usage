from time import perf_counter

from numpy import ndarray, zeros

from model import Model


class VS:
	def __init__(
			self, model: Model, target: ndarray, record: bool = False, verbose: bool = False
			):

		self.model = model
		self.result = zeros( self.model.actuation.shape )

		self.record = record
		if self.record:
			self.times = [ ]

		self.verbose = verbose

	def apply_result( self ):
		self.model.actuation = self.result

	def compute_actuation( self ):

		if self.record:
			ti = perf_counter()

		# TODO: Claire implement actuation with visual servoing method using self.model.state

		if self.record:
			self.times.append( perf_counter() - ti )
