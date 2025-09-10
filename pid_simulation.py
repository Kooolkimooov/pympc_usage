from numpy import array, eye, pi, set_printoptions, zeros, sqrt

from pympc.controllers.pid import PID
from pympc.models.dynamics.bluerov import BluerovXYZPsi as Bluerov
from pympc.models.model import Model
from pympc.utils import Logger

import matplotlib.pyplot as plt

set_printoptions( precision=2, linewidth=10000, suppress=True )

dynamics = Bluerov( reference_frame='NED', water_surface_depth=0., water_current=array([sqrt(2.0), 0.0, 0.0]) )

time_step = 0.1

initial_actuation = zeros( (dynamics.actuation_size,) )

initial_state = zeros( (dynamics.state_size,) )
initial_state[ dynamics.position[ 0 ] ] = 2.
initial_state[ dynamics.position[ 2 ] ] = 1.

model = Model(
        dynamics=dynamics,
        time_step=time_step,
        initial_state=initial_state,
        initial_actuation=initial_actuation,
        record=True
)

proportional = eye( dynamics.state_size // 2 )[ dynamics.six_dof_actuation_mask, : ]
integral = eye( dynamics.state_size // 2 )[ dynamics.six_dof_actuation_mask, : ]
derivative = eye( dynamics.state_size // 2 )[ dynamics.six_dof_actuation_mask, : ]
offset = zeros( (dynamics.actuation_size,) )

proportional[ dynamics.linear_actuation[ :2 ] ] *= 80.0
proportional[ dynamics.linear_actuation[ 2 ] ] *= 30.0
proportional[ dynamics.angular_actuation ] *= 1.1

integral[ dynamics.linear_actuation[ :2 ] ] *= 30.0
integral[ dynamics.linear_actuation[ 2 ] ] *= 0.0
integral[ dynamics.angular_actuation ] *= 0.0

derivative[ dynamics.linear_actuation[:2] ] *= 5500.0
derivative[ dynamics.linear_actuation[2] ] *= 1000.0
derivative[ dynamics.angular_actuation ] *= 1.0

offset[ dynamics.linear_actuation[ 2 ] ] = 18.25

pid = PID(
        model=model,
        target=array( [ 3.0, 0.0, 2.0, 0.0, 0.0, 0.0 ] ),
        proportional=proportional,
        integral=integral,
        derivative=derivative,
        offset=offset,
        anti_windup_limit=1e2,
        record=False
)

logger = Logger()

logger.lognl( 'target\tposition\terror\tactuation' )

for i in range( 100 ):
    model.actuation = pid.step()
    model.step()

    logger.log( f'{pid.target}' )
    logger.log( f'{model.state[ :dynamics.state_size // 2 ]}' )
    logger.log( f'{pid.error}' )
    logger.lognl( f'{model.actuation}' )

plt.plot(array(model.previous_states)[:, :3])
plt.show()
