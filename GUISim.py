import tkinter as tk, threading
from tkinter import ttk
import numpy as np
import os, imageio
from PIL import Image, ImageTk
from main import SimInterface
import queue


datafolder = '\\'.join(os.getcwd().split('\\')[:])+'\\simulation_results'
files = os.listdir(datafolder)

lastfile = datafolder + '\\' + files[-1]


TIME = 5000
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

Sim.set_fire(0.1, (150, 150))

Sim.clear_space((20, 240), (40, 30))


class Window(tk.Frame):

    def __init__(self, master=None, sim = Sim):

        self.simulation = Sim

        self.framedata = None
        self.t_frame = 0

        self.gif = False
        self.mp4 = False

        self.frame_counter = 0
        self.retardant_mode = False

        self.mouse_position = [0, 0]

        self.mouse_positions = {
            'old': [0, 0],
            'new': [0, 0]
        }

        self.click_counter = 0

        self.queue = queue.Queue()

        tk.Frame.__init__(self, master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=1)

        self.master.bind('<Motion>', self.__mouse_motion)

        self.label = tk.Label(self.master)
        self.label.pack(side=tk.LEFT, anchor=tk.SW)

        self.retardant = tk.Button(self.master, text='Retardant', command=self.__change_retardant_mode)
        self.retardant.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.NW, expand=tk.YES)

        self.play = tk.Button(self.master, text='Play', command=self.next_step)
        self.play.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.NW, expand=tk.YES)

        self.stop = tk.Button(self.master, text='Stop', command=self.__stop_callback())
        self.stop.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.NW, expand=tk.YES)

        self.framesvar = tk.StringVar()
        self.framesvar.set(str(self.frame_counter))

        self.framecount = tk.Entry(self.master, textvariable=self.framesvar)
        self.framecount.bind("<Return>", lambda event: self.__entry_callback())
        self.framecount.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.NW, expand=tk.YES)

        self.timevar = tk.StringVar()
        self.time = tk.Label(self.master, textvariable=self.timevar)
        self.timevar.set(self.t_frame)
        self.time.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.NW, expand=tk.YES)

        self.save = tk.Button(self.master, text='Save', command=self.save)
        self.save.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.NW, expand=tk.YES)

        self.__frame_update(self.label)

        self.master.bind('<Motion>', self.__mouse_motion)
        self.master.bind('<Button-1>', self.__mouse_click)

    def __mouse_motion(self, event):

        if self.retardant_mode:

            x = root.winfo_pointerx() - root.winfo_rootx()
            y = root.winfo_pointery() - root.winfo_rooty()

            self.mouse_position = [y, x]

        else:
            pass

    def __mouse_click(self, event):

        if self.retardant_mode:

            if 0 <= self.mouse_position[0] < GRID_SIZE[0] and 0 <= self.mouse_position[1] < GRID_SIZE[1]:
                print(self.mouse_position)
                new_old = list(self.mouse_positions['new'])

                self.mouse_positions['new'] = self.mouse_position
                self.mouse_positions['old'] = new_old

                if self.click_counter < 1:
                    self.click_counter += 1

                else:
                    print('Dropping')
                    drop ={
                        'time': self.t_frame+1,              # Frame at which to drop
                        'velocity': 12,         # Pixels/frame
                        'amount': 1.5,            # Average amount per cell; randomisation applies
                        'start': self.mouse_positions['old'],      # Starting point of the drop
                        'end': self.mouse_positions['new'],       # Ending point of the drop
                        'width': 7              # Width of the retardant line (only uneven numbers work)
                    }

                    self.simulation.plan_retardant_drops(drop1=drop)
                    self.click_counter = 0

            else:
                pass

        else:
            pass

    def __entry_callback(self):

        try:
            self.frame_counter = int(self.framesvar.get())

        except TypeError:
            self.frame_counter = self.framesvar.get()
        print("Iterations: ", self.frame_counter)
        return True

    def __change_retardant_mode(self):

        if self.retardant_mode:
            self.retardant_mode = False

            self.mouse_position = {
            'old': [0, 0],
            'new': [0, 0]
            }

        else:
            self.retardant_mode = True

        print("RETARDANT MODE: ", self.retardant_mode)

    def __stop_callback(self):

        self.frame_counter = 0

    def __update(self):

        self.t_frame += 1
        self.timevar.set(self.t_frame)
        self.simulation.update(self.t_frame)
        self.__frame_update(self.label)

    def __counter_update(self):

        self.__update()
        self.frame_counter -= 1
        self.framesvar.set(str(self.frame_counter))

    def next_step(self):

        if self.frame_counter > 0:

            while self.frame_counter > 0:
                self.__counter_update()

        else:
            self.__update()

    def save(self):

        self.simulation.simulation.save_simulation()

    def __frame_update(self, label):

        image = ImageTk.PhotoImage(Image.fromarray(self.simulation.simulation.coloured[self.t_frame, :, :, :]))
        label.config(image=image)
        label.image = image

    def __get_frame(self, t: int):
        return self.framedata[t]




root = tk.Tk()
app = Window(root)
root.wm_title("Tkinter window")
root.geometry(f"{GRID_SIZE[0]+50}x{GRID_SIZE[1]}")
root.mainloop()
