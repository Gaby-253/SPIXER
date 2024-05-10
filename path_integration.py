import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

import pygenn
from pygenn import GeNNModel
from pygenn import GeNNModel, init_postsynaptic, init_sparse_connectivity, init_var, init_weight_update
from pygenn import (create_neuron_model, create_current_source_model,
                    init_postsynaptic, init_weight_update, GeNNModel, create_var_ref)

import utils


class PI:

    def __init__(self, display=False):
        self.model = GeNNModel("float", "tutorial1", backend="single_threaded_cpu")
        self.model.dt = 0.1

        self.n_TN2 = 2 # left and right linear velocity
        self.n_TL = 16 # heading
        self.n_TB1 = 8 # ring attractor
        self.n_Generator = 1
        self.n_CPU4 = 16
        self.n_CPU1a = 14
        self.n_CPU1b = 2
        self.n_motor = 2

        self.lif_params = {"C": 0.2, "TauM": 20.0,
                    "Vrest": -70.0, "Vreset": -70.0, "Vthresh": -45.0, "Ioffset": 0.0,"TauRefrac": 2.0}
        self.lif_init = {"V": -52.0, "RefracTime": 0.0}

        ###########################
        ### WEIGHTS DEFINITIONS ###
        ###########################

        # CL1 and Pontine are omitted for simplicity

        self.TLtoTB1_init = 1.-0.33
        self.TB1toTB1_init = 0.33
        self.TB1toCPU4_init = -1
        self.TB1toCPU1a_init = -1
        self.TB1toCPU1b_init = -1
        self.CPU4toCPU1a_init = 1.
        self.CPU4toCPU1b_init = 1.


        self.TLtoTB1 = np.concatenate([np.diag([1]*self.n_TB1), np.diag([1]*self.n_TB1)], axis=1) * self.TLtoTB1_init

        num_neurons = self.n_TB1 = 8  # You can adjust this number based on your specific model
        theta = np.linspace(0, 2 * np.pi, num_neurons, endpoint=False)# Generate the theta values for each neuron
        self.TB1toTB1 = np.zeros((num_neurons, num_neurons)) # Create the weight matrix
        for i in range(num_neurons):
            for j in range(num_neurons):
                self.TB1toTB1[i, j] = (np.cos(theta[i] - theta[j]) - 1) / 2 # Calculate the weight using the provided formula
        self.TB1toTB1 *= self.TB1toTB1_init

        ###########################
        ### NEURONS DEFINITIONS ###
        ###########################

        CPU4LIF = pygenn.create_neuron_model(
            "CPU4LIF",

            sim_code=
                """
                V += (-V + Isyn) * (dt / tau);
                wgen = wgen + IsynTN + IsynTB1;
                """,
            threshold_condition_code="V >= 1.0",
            reset_code=
                """
                V = 0.0;
                """,

            params=["tau"],
            vars=[("V", "scalar", pygenn.VarAccess.READ_WRITE), ("wgen", "scalar", pygenn.VarAccess.READ_WRITE)], additional_input_vars=[("IsynTN", "scalar", 0.), ("IsynTB1", "scalar", 0.)])

        self.cpu4_init = {"V": -52.0, "RefracTime": 0.0, "wgen": 0.0}

        self.TN2 = self.model.add_neuron_population("TN", self.n_TN2, "LIF", self.lif_params, self.lif_init)
        self.TL = self.model.add_neuron_population("TL", self.n_TL, "LIF", self.lif_params, self.lif_init)
        self.TB1 = self.model.add_neuron_population("TB1", self.n_TB1, "LIF", self.lif_params, self.lif_init)
        self.CPU4 = self.model.add_neuron_population("CPU4", self.n_CPU4, CPU4LIF, {"tau": 20.0}, {"V":  0.0, "wgen": 0.0})
        self.CPU1a = self.model.add_neuron_population("CPU1a", self.n_CPU1a, "LIF", self.lif_params, self.lif_init)
        self.CPU1b = self.model.add_neuron_population("CPU1b", self.n_CPU1b, "LIF", self.lif_params, self.lif_init)
        self.motor = self.model.add_neuron_population("Motor", self.n_motor, "LIF", self.lif_params, self.lif_init)
        self.Generator = self.model.add_neuron_population("Generator", self.n_Generator, "LIF", self.lif_params, self.lif_init)

        self.TN2.spike_recording_enabled = True
        self.TL.spike_recording_enabled = True
        self.TB1.spike_recording_enabled = True
        self.CPU4.spike_recording_enabled = True
        self.CPU1a.spike_recording_enabled = True
        self.CPU1b.spike_recording_enabled = True
        self.motor.spike_recording_enabled = True
        self.Generator.spike_recording_enabled = True

        ###########################
        ### SYNAPSE DEFINITIONS ###
        ###########################

        self.NMDA_post_syn_params = {"tau": 100.0}
        self.GABAA_post_syn_params = {"tau": 50.0}

        num_directions = self.n_TB1
        tmp = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
        self.TL_pref_azimuth = list(tmp) + list(tmp)


        TN2toCPU4 = pygenn.create_weight_update_model(
            "TN2toCPU4",

            pre_spike_syn_code=
                """
                addToPost(0.000001);
                """)

        TB1toCPU4 = pygenn.create_weight_update_model(
            "TB1toCPU4",

            pre_spike_syn_code=
                """
                addToPost(-0.0000001);
                """)

        GeneratortoCPU4 = pygenn.create_weight_update_model(
            "GeneratortoCPU4",

            post_neuron_var_refs=[("wgen_post", "scalar")],

            pre_spike_syn_code=
                """
                addToPost(wgen_post);
                """)


        self.model.add_synapse_population("TL_TB1", "DENSE",
            self.TL, self.TB1,
            init_weight_update("StaticPulse",{}, {"g": self.TLtoTB1.transpose().ravel()}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))

        self.model.add_synapse_population("TB1_TB1", "DENSE",
            self.TB1, self.TB1,
            init_weight_update("StaticPulse",{}, {"g": self.TB1toTB1.transpose().ravel()}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))

        tmp = self.model.add_synapse_population("TB1_CPU1a", "SPARSE",
            self.TB1, self.CPU1a,
            init_weight_update("StaticPulse",{}, {"g": self.TB1toCPU1a_init}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections(list(range(1,8)) +list(range(0,7)), list(range(0,14)))

        tmp = self.model.add_synapse_population("TB1_CPU1b", "SPARSE",
            self.TB1, self.CPU1b,
            init_weight_update("StaticPulse",{}, {"g": self.TB1toCPU1b_init}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections([7,0], [0,1])

        tmp = self.model.add_synapse_population("TB1_CPU4", "SPARSE",
            self.TB1, self.CPU4,
            init_weight_update(TB1toCPU4,{},  {}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections(list(range(0,8))+list(range(0,8)), list(range(0,16)))
        tmp.post_target_var = "IsynTB1"

        tmp = self.model.add_synapse_population("TN2_CPU4", "SPARSE",
            self.TN2, self.CPU4,
            init_weight_update(TN2toCPU4,{},  {}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections(list([0]*8 +[1]*8), list(range(0,16)))
        tmp.post_target_var = "IsynTN"

        tmp = self.model.add_synapse_population("Generator_CPU4", "DENSE",
            self.Generator, self.CPU4,
            init_weight_update(GeneratortoCPU4,  post_var_refs={"wgen_post": create_var_ref(self.CPU4, "wgen")}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.post_target_var = "Isyn"


        tmp = self.model.add_synapse_population("CPU1a_Motor", "SPARSE",
            self.CPU1a, self.motor,
            init_weight_update("StaticPulse",{}, {"g": 0.005}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections(list(range(0,14)), [1]*7 + [0]*7)

        tmp = self.model.add_synapse_population("CPU1b_Motor", "SPARSE",
            self.CPU1b, self.motor,
            init_weight_update("StaticPulse",{}, {"g": 0.005}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections([0,1], [0,1])


        tmp = self.model.add_synapse_population("CPU4_CPU1a", "SPARSE",
            self.CPU4, self.CPU1a,
            init_weight_update("StaticPulse",{}, {"g": self.CPU4toCPU1a_init}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections([1,2,3,4,5,6,7, 8,9,10,11,12,13,14], [7,8,9,10,11,12,13, 0,1,2,3,4,5,6] )

        tmp = self.model.add_synapse_population("CPU4_CPU1b", "SPARSE",
            self.CPU4, self.CPU1b,
            init_weight_update("StaticPulse",{}, {"g": self.CPU4toCPU1b_init}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections([0,15], [0,1])

        tmp = self.model.add_synapse_population("CPU4_CPU1a_INH", "SPARSE",
            self.CPU4, self.CPU1a,
            init_weight_update("StaticPulse",{}, {"g": -self.CPU4toCPU1a_init}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections([0,1,2,3, 5,6,7, 8,9,10, 12,13,14,15], [10,11,12,13, 7,8,9, 4,5,6, 0,1,2,3] )

        tmp = self.model.add_synapse_population("CPU4_CPU1b_INH", "SPARSE",
            self.CPU4, self.CPU1b,
            init_weight_update("StaticPulse",{}, {"g": -self.CPU4toCPU1b_init}),
            init_postsynaptic("ExpCurr", self.NMDA_post_syn_params))
        tmp.set_sparse_connections([4,11], [0,1] )


        self.cs_model = create_current_source_model(
            "cs_model",
            vars=[("magnitude", "scalar")],
            injection_code="injectCurrent(magnitude);")

        self.TL_input = self.model.add_current_source("TL_input", self.cs_model,
                                                self.TL, {}, {"magnitude": 0.0})

        self.TN2_input = self.model.add_current_source("TN2_input", self.cs_model,
                                                self.TN2, {}, {"magnitude": 0.0})

        self.Generator_input = self.model.add_current_source("Generator_input", self.cs_model,
                                                self.Generator, {}, {"magnitude": 0.0})

        self.model.build()
        print("model built")
        self.model.load(num_recording_timesteps=10000)




    def move(self, azimuth, speedl, speedr, generator_cur=0.3):
        """
        Function to be called at every timestep of the robot
        Integrate the distance traveled during one timestep in the PI model,
        the activity of the left and right motor neurons is returned. The
        preferred direction of the model can be inferred by computing the difference
        between the left and right motor neurons number of spikes.

        azimuth:
        speedl: linear speed measured by the left TN neuron
        speedr: linear speed measured by the right TN neuron (orthogonal to the left one)

        Generate one second of GeNN simulation (which is not 1 second in real time world, probably much less)
        The simulation worked with generator_cur=0.3

        Returns the activity of the left and right motor neurons (number of spikes)
        recorded during a full second of GeNN simulation (which is not 1 second in real time world, probably much less)
        """

        self.TL_input.vars["magnitude"].values = np.cos(self.TL_pref_azimuth - np.radians(azimuth))
        self.TL_input.vars["magnitude"].push_to_device()
        self.TN2_input.vars["magnitude"].values = np.array([speedl, speedr])
        self.TN2_input.vars["magnitude"].push_to_device()
        self.Generator_input.vars["magnitude"].values = generator_cur
        self.Generator_input.vars["magnitude"].push_to_device()
        for step in range(10000):
            self.model.step_time()

        spike_times, spike_ids = self.motor.spike_recording_data[0]
        n_left = (np.array(spike_ids) == 0).sum()
        n_right = (np.array(spike_ids) == 1).sum()

        # Number of spikes on the left / right motor neurons
        return (n_left, n_right)

    def display_raster(self):
        neurons = [self.TN2, self.TL, self.TB1, self.CPU4, self.CPU1a, self.CPU1b, self.motor, self.Generator]

        spike_times, spike_ids = self.motor.spike_recording_data[0]
        n_left = (np.array(spike_ids) == 0).sum()
        n_right = (np.array(spike_ids) == 1).sum()
        print("N spikes motor LEFT: ", n_left)
        print("N spikes motor RIGHT: ", n_right)

        labels = ["TN2 (left and right velocity) n=2",
                    "TL (Landmark + Polarization) n=16",
                    "TB1 (Ring attractor with lateral inhib) n=8",
                    "CPU4",
                    "CPU1a",
                    "CPU1b",
                    "Motor",
                    "Generator"]
        # Create figure with one axis per neuron population
        fig, axes = plt.subplots(len(neurons), sharex=True, figsize=(12,12))

        # Loop through neuron populations and the axis we're going to plot their raster plot on
        for i, (n, a) in enumerate(zip(neurons, axes)):
            # Extract spike times and IDs and plot

            spike_times, spike_ids = n.spike_recording_data[0]
            a.scatter(spike_times, spike_ids, s=1)

            a.set_title(labels[i])
            a.set_ylabel("Neuron ID")
            a.set_ylim((-1, n.num_neurons))


    def simulatepi(self, n_steps_forward=50, n_steps_return=60, tortuosity=0.4):

        def calculate_azimuth(x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            azimuth_rad = math.atan2(dy, dx)  # Calculate azimuth in radians
            azimuth_deg = math.degrees(azimuth_rad)  # Convert to degrees
            return (azimuth_deg + 360) % 360  # Normalize to 0-359 degrees


        # Initialize parameters for the walk
        kappa = tortuosity  # concentration parameter (higher means less spread)
        bias_direction = np.random.choice([-1, 1])  # -1 for left, 1 for right
        bias_strength = np.radians(1.6)  # strength of the bias per step in radians

        # Initialize starting point and base direction
        x, y = 0, 0
        coordinates_outbound = [(x, y)]
        base_direction = np.radians(np.random.randint(360))  # initial random direction in radians

        plt.figure(figsize=(8, 8))
        # Generate outbound trajectory
        for i in range(n_steps_forward):

            if i % 25 == 0:
                tmp = np.random.randint(6)
                bias_direction = np.random.choice([-1, 1])  # -1 for left, 1 for right
                bias_strength = np.radians(tmp)  # strength of the bias per step in radians


            base_direction += bias_strength * bias_direction
            step_angle = np.random.vonmises(base_direction, kappa)
            dx = np.cos(step_angle)
            dy = np.sin(step_angle)
            x += dx
            y += dy
            coordinates_outbound.append((x, y))
            if i == 0:
                plt.quiver(x - dx, y - dy, dx, dy, angles='xy', scale_units='xy', scale=0.4, width=0.002,  color='blue', label='Outbound Trajectory')
            else:
                plt.quiver(x - dx, y - dy, dx, dy, angles='xy', scale_units='xy', scale=0.4, width=0.002,  color='blue')

        azimuths = []
        for i in range(1, len(coordinates_outbound)):
            x1, y1 = coordinates_outbound[i-1]
            x2, y2 = coordinates_outbound[i]
            azimuth = calculate_azimuth(x1, y1, x2, y2)
            azimuths.append(azimuth)

        for az in azimuths:
            self.move(az, speedl=0.5, speedr=0.5, generator_cur=0.3)

        azimuth = azimuths[-1]
        azimuths = []
        for timestep in range(n_steps_return):


            spike_times, spike_ids = self.motor.spike_recording_data[0]
            n_left = (np.array(spike_ids) == 0).sum()
            n_right = (np.array(spike_ids) == 1).sum()
            # n_left = np.random.randint(10)
            # n_right = np.random.randint(10)
            if n_left > n_right:
                azimuth -= 30
            else:
                azimuth += 30
            azimuth = azimuth%360

            #self.move(azimuth, speedl=0.5, speedr=0.5, generator_cur=0.3)
            self.move(azimuth, speedl=0.5, speedr=0.5, generator_cur=0.3)
            azimuths.append(azimuth)

        def generate_coordinates_from_azimuths(start_x, start_y, azimuths):
            coordinates = [(start_x, start_y)]  # Initialize with the starting position
            current_x, current_y = start_x, start_y

            for azimuth in azimuths:
                # Convert azimuth from degrees to radians
                azimuth_rad = math.radians(azimuth)
                # Compute the delta x and delta y
                delta_x = math.cos(azimuth_rad)
                delta_y = math.sin(azimuth_rad)
                # Update the current position
                current_x += delta_x
                current_y += delta_y
                # Append the new position to the list of coordinates
                coordinates.append((current_x, current_y))

            return coordinates
        coordinates_return = generate_coordinates_from_azimuths(x, y, azimuths)

        plt.quiver(*zip(*coordinates_return[:-1]), np.diff([x[0] for x in coordinates_return]), np.diff([y[1] for y in coordinates_return]), angles='xy', scale_units='xy', scale=0.34, width=0.002, color="green", label='Return Trajectory')

        plt.plot(coordinates_outbound[0][0], coordinates_outbound[0][1], marker='o', color='red', label='Nest')

        # Labels and legends
        plt.title('Biased Trajectory Simulation with Outbound and Return Paths')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

        # Plotting
        plt.figure(figsize=(8, 8))
        # Outbound trajectory
        x_coords, y_coords = zip(*coordinates_outbound)
        plt.plot(x_coords, y_coords, marker='', linestyle='-', color='blue', label='Outbound Trajectory')
        # Return trajectory
        x_coords_return, y_coords_return = zip(*coordinates_return)
        plt.plot(x_coords_return, y_coords_return, marker='', linestyle='-', color="green", label='Return Trajectory')
        # Starting point
        plt.plot(coordinates_outbound[0][0], coordinates_outbound[0][1], marker='o', color='red', label='Nest')

        # Labels and legends
        plt.title('Biased Trajectory Simulation with Outbound and Return Paths')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
