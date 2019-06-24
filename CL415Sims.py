from main import SimInterface
import numpy as np

TIME = 500
GRID_SIZE = (200, 400)

Sim = SimInterface(TIME, GRID_SIZE,
                     maxfuel=4,
                     evaporation=0.00075,
                     efficiency=0.75,
                     wind={
                         'direction': [0, 1],
                         'magnitude': 0.75,
                         'randomness': 0.4
                     },
                     elevation={
                         'direction': [0, 1],
                         'slope_angle': 10 * np.pi / 180,
                         'randomness': 0.5
                     }
                )


for i in range(GRID_SIZE[0]):
    Sim.set_fire(0.1, (i, np.random.randint(0, 5)))

Sim.plan_retardant_drops(
        drop_1={
            'time': 140,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [GRID_SIZE[0], 160],      # Starting point of the drop
            'end': [100-10, 160],       # Ending point of the drop
            'width': 7              # Width of the retardant line (only uneven numbers work)
        },
        drop_2={
            'time': 141,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [2*GRID_SIZE[0]//3, 166],      # Starting point of the drop
            'end': [-10, 166],       # Ending point of the drop
            'width': 7              # Width of the retardant line (only uneven numbers work)
        },
        drop_3={
            'time': 163,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [GRID_SIZE[0], 170],      # Starting point of the drop
            'end': [100-10, 170],       # Ending point of the drop
            'width': 7              # Width of the retardant line (only uneven numbers work)
        },
        drop_4={
            'time': 164,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [2*GRID_SIZE[0]//3, 176],      # Starting point of the drop
            'end': [-10, 176],       # Ending point of the drop
            'width': 7              # Width of the retardant line (only uneven numbers work)
        }
)

Sim.run()