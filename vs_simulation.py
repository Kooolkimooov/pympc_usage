from pympc.controllers.vs import VS
from numpy import zeros, array, set_printoptions

set_printoptions( precision=2, linewidth=10000, suppress=True )

leader = zeros((6, 1))
follower = zeros((6, 1))

leader[0] = 0.5

desired_feature = array([1, 0, 0.5])

print(f'{leader=}')
print(f'{follower=}')
print(f'{desired_feature=}')

vs = VS(leader, follower, desired_feature, cable_length=1.0, verbose=True)

actuation = vs.compute_actuation()

print(f'{actuation=}')