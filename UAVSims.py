from main import SimInterface
import numpy as np

TIME = 400
GRID_SIZE = (200, 400)

Sim = SimInterface(TIME, GRID_SIZE,
                     maxfuel=4,
                     evaporation=0.00075,
                     efficiency=1.5,
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
            'time': 110,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.25,            # Average amount per cell; randomisation applies
            'start': [GRID_SIZE[0], 120],      # Starting point of the drop
            'end': [100-10, 120],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_2={
            'time': 113,              # Frame at which to drop
            'velocity': 10,           # Pixels/frame
            'amount': 1.25,            # Average amount per cell; randomisation applies
            'start': [2*GRID_SIZE[0]//3, 122],      # Starting point of the drop
            'end': [-1, 122],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_3={
            'time': 116,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [GRID_SIZE[0], 124],      # Starting point of the drop
            'end': [100-10, 124],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_4={
            'time': 119,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [2*GRID_SIZE[0]//3, 126],      # Starting point of the drop
            'end': [-1, 126],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_5={
            'time': 150,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [GRID_SIZE[0], 170],      # Starting point of the drop
            'end': [100-10, 170],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_6={
            'time': 153,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [2*GRID_SIZE[0]//3, 175],      # Starting point of the drop
            'end': [-10, 175],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_7={
            'time': 156,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [GRID_SIZE[0], 172],      # Starting point of the drop
            'end': [100-10, 172],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_8={
            'time': 159,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [2*GRID_SIZE[0]//3, 177],      # Starting point of the drop
            'end': [-10, 177],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        }
)

Sim.run()