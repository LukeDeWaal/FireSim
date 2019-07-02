from main import SimInterface
import numpy as np

TIME = 120
GRID_SIZE = (300, 300)

Sim = SimInterface(
    TIME,
    GRID_SIZE,
    maxfuel=4,
    evaporation=0.00095,
    efficiency=1.5,
    wind={
        'direction': [1, -1],
        'magnitude': 0.75,
        'randomness': 0.4
    },
    elevation={
        'direction': [-1, 1],
        'slope_angle': 12 * np.pi / 180,
        'randomness': 0.5
    }
)

Sim.clear_space((10, 250), (20, 40))

Sim.set_fire(0.1, (GRID_SIZE[0]//2 - 30, GRID_SIZE[1]//2 + 30))
for i in range(5):
    Sim.set_fire(0.1, (GRID_SIZE[0]//2 + np.random.randint(-7, 7) - 30, GRID_SIZE[1]//2 + np.random.randint(-7, 7) + 30))

Sim.plan_retardant_drops(
        drop_1={
            'time': 60,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.25,            # Average amount per cell; randomisation applies
            'start': [100, 130],      # Starting point of the drop
            'end': [140, 120],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_2={
            'time': 55,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.25,            # Average amount per cell; randomisation applies
            'start': [140, 120],     # Starting point of the drop
            'end': [170, 150],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_3={
            'time': 50,  # Frame at which to drop
            'velocity': 10,  # Pixels/frame
            'amount': 1.25,  # Average amount per cell; randomisation applies
            'start': [170, 150],  # Starting point of the drop
            'end': [170, 190],  # Ending point of the drop
            'width': 5  # Width of the retardant line (only uneven numbers work)
        },
        drop_4={
            'time': 60,  # Frame at which to drop
            'velocity': 10,  # Pixels/frame
            'amount': 1.25,  # Average amount per cell; randomisation applies
            'start': [100, 130],  # Starting point of the drop
            'end': [140, 120],  # Ending point of the drop
            'width': 5  # Width of the retardant line (only uneven numbers work)
        },
        drop_5={
            'time': 55,  # Frame at which to drop
            'velocity': 10,  # Pixels/frame
            'amount': 1.25,  # Average amount per cell; randomisation applies
            'start': [140, 120],  # Starting point of the drop
            'end': [170, 150],  # Ending point of the drop
            'width': 5  # Width of the retardant line (only uneven numbers work)
        },
        drop_6={
            'time': 50,  # Frame at which to drop
            'velocity': 10,  # Pixels/frame
            'amount': 1.25,  # Average amount per cell; randomisation applies
            'start': [170, 150],  # Starting point of the drop
            'end': [170, 190],  # Ending point of the drop
            'width': 5  # Width of the retardant line (only uneven numbers work)
        }
)

Sim.run()