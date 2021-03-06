import tempfile
from typing import Dict, Optional, Type, Iterable
# import time as clock

import torch
import dwave_qbsolv as qbs

from .monitors import AbstractMonitor
from .nodes import Nodes
from .topology import AbstractConnection
from ..learning.reward import AbstractReward


# While classical code would indicate a spike at 0 = thresh - v (as the equation is v >= thresh),
# whether Quantum Annealer would do this is random, as the energy value would normally be = 0 here.
# So to make sure it indicates a spike, we need to give it a nudging value whose absolute value is smaller than
# the smallest "actual" changes to the values in our equation that occur each timestep,
# which should be the changes to the weights (which are largely determined by the decay of the Inputs)
# Problem: Smallest changes are at approximately the fourth digit after the point,
# while Quantum Annealers at the moment have a precision of 3 digits -> No effect? -> Thomas said we don't need it
# NUDGE = -0.0001


def load(file_name: str, map_location: str = "cpu", learning: bool = None) -> "Network":
    # language=rst
    """
    Loads serialized network object from disk.

    :param file_name: Path to serialized network object on disk.
    :param map_location: One of ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :param learning: Whether to load with learning enabled. Default loads value from
        disk.
    """
    network = torch.load(open(file_name, "rb"), map_location=map_location)
    if learning is not None and "learning" in vars(network):
        network.learning = learning

    return network


class Network(torch.nn.Module):
    # language=rst
    """
    Central object of the ``bindsnet_qa`` package. Responsible for the simulation and
    interaction of nodes and connections.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet_qa         import encoding
        from bindsnet_qa.network import Network, nodes, topology, monitors

        network = Network(dt=1.0)  # Instantiates network.

        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])

        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')

        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(100)  # Generate random Poisson rates for 100 input neurons.
        train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.

        # Simulate network on generated spike trains.
        inputs = {'X' : train}  # Create inputs mapping.
        network.run(inputs=inputs, time=5000)  # Run network simulation.

        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')

        plt.tight_layout(); plt.show()
    """

    def __init__(
        self,
        dt: float = 1.0,
        batch_size: int = 1,
        learning: bool = True,
        reward_fn: Optional[Type[AbstractReward]] = None,
    ) -> None:
        # language=rst
        """
        Initializes network object.

        :param dt: Simulation timestep.
        :param batch_size: Mini-batch size.
        :param learning: Whether to allow connection updates. True by default.
        :param reward_fn: Optional class allowing for modification of reward in case of
            reward-modulated learning.
        """
        super().__init__()

        self.dt = dt
        self.batch_size = batch_size

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

        if reward_fn is not None:
            self.reward_fn = reward_fn()
        else:
            self.reward_fn = None

    def add_layer(self, layer: Nodes, name: str) -> None:
        # language=rst
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)
        layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(
        self, connection: AbstractConnection, source: str, target: str
    ) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target)] = connection
        self.add_module(source + "_to_" + target, connection)

        connection.dt = self.dt
        connection.train(self.learning)

    def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
        # language=rst
        """
        Adds a monitor on a network object to the network.

        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        # language=rst
        """
        Serializes the network object to disk.

        :param file_name: Path to store serialized network object on disk.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet_qa.network import *
            from bindsnet_qa.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
        """
        torch.save(self, open(file_name, "wb"))

    def clone(self) -> "Network":
        # language=rst
        """
        Returns a cloned network object.
        
        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)

    def _get_inputs(self, layers: Iterable = None) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """
        inputs = {}

        if layers is None:
            layers = self.layers

        # Loop over network connections.
        for c in self.connections:
            if c[1] in layers:
                # Fetch source and target populations.
                source = self.connections[c].source
                target = self.connections[c].target

                if not c[1] in inputs:
                    inputs[c[1]] = torch.zeros(
                        self.batch_size, *target.shape, device=target.s.device
                    )

                # Add to input: source's spikes multiplied by connection weights.
                inputs[c[1]] += self.connections[c].compute(source.s)

        return inputs

    def penalty_one_spike(self, layer: str) -> float:
        # language=rst
        """
        Calculates the penalty-value for the Quantum Annealer which is used to prevent a layer from having
        two spiking nodes. To be called if the layers attribute one_spike is true.

        :return: a float value to be used in Quantum Annealing to keep a certain node from spiking if another node in
            its layer spikes
        """
        penalty = 1
        # for every incoming connection
        for c in self.connections:
            if c[1] == layer:
                c_v = self.connections[c]
                if c_v.wmax > 0:
                    # increase punishment by the maximum weight * number of incoming connections
                    penalty += (c_v.wmax * c_v.source.n)
                # max_bias = torch.max(c_v.b)
                # if max_bias > 0: #never the case: is always bias = 0 in our example
                     # if that makes a positive impact for QA, add bias
                     # penalty += max_bias
        # punishment is now bigger than the cumulative value of all "new" inputs to the layer in this timestep
        # 2* because is needed for "row node" as well as "column node"
        return 2 * penalty

    def reward_inhibitory(self, layer: str) -> float:
        # language=rst
        """
        Calculates the reward-value for the Quantum Annealer which is used to prevent a layer from "ignoring" inhibitory
        inputs. To be calculated for inhibitory layer.

        :return: a float value to be used in Quantum Annealing to clamp a certain inhibitory node's qubit, whose node
            spiked in the last time window (and is now in refractory period), to 1
        """
        reward = -1
        # for every outgoing connection
        for c in self.connections:
            if c[0] == layer:
                c_v = self.connections[c]
                if c_v.wmin < 0:
                    # increase reward by the minimum weight * number of outgoing connections
                    reward += (c_v.wmin * c_v.target.n)
        # reward is now bigger than the cumulative value of all "new" outputs from the layer in this timestep
        return reward

    def forward_qa(self, penalties_and_rewards: dict, num_repeats: int) -> None:
        # language=rst
        """
        Runs a single simulation step.
        Only works for batch_size = 1 and DiehlAndCook-Network.
        """
        conn = self.connections
        inputs = {}  # shape: number of layers * number of neurons
        qubo = {}  # shape: (number of neurons * number of layers)^2
        encoding = {}  # to remember at which row_nr which layer starts, shape: number of layers
        nr_of_prev_nodes = 0

        for l in self.layers:
            l_v = self.layers[l]
            inputs[l] = torch.zeros(1, l_v.n)  # a tensor for each layer
            encoding[l] = nr_of_prev_nodes
            nr_of_prev_nodes += l_v.n

            # Decay voltages.
            if l == 'Ae':  # layer of DiehlAndCookNodes
                l_v.v = l_v.decay * (l_v.v - l_v.rest) + l_v.rest
                # and adaptive thresholds
                l_v.theta *= l_v.theta_decay
            elif l == 'Ai':  # layer of LIF-Nodes
                l_v.v = l_v.decay * (l_v.v - l_v.rest) + l_v.rest
            # else: l == 'X' -> layer of Input-Nodes -> does not have a voltage


        # prepare Quantum Annealing and get inputs
        # go through layers (and corresponding Input-connections)
        l_ae_v = self.layers['Ae']
        l_ai_v = self.layers['Ai']

        # Layer X of Input-Neurons: nothing to do

        # Layer Ae of excitatory DiehlAndCook-Nodes
        # needed later for Inputs from layers X and Ai (connections X->Ae and Ai->Ae)
        s_view_x = self.layers['X'].s.float().view(self.layers['X'].s.size(0), -1)[0].tolist()
        s_view_ai = self.layers['Ai'].s.float().view(self.layers['Ai'].s.size(0), -1)[0].tolist()

        for node_ae in range(l_ae_v.n):
            # Could spike -> needs constraints
            if l_ae_v.refrac_count[0, node_ae].item() == 0:
                nr_ae = node_ae + encoding['Ae']

                # diagonal
                # = threshold + variable threshold theta - (voltage + connection-bias)
                # for c in conn: not needed in this case: always bias = 0
                    # if c[1] == 'Ae':
                        # l_ae_v.v[0, node] += conn[c].b[node]
                qubo[(nr_ae, nr_ae)] = l_ae_v.thresh.item() + l_ae_v.theta[node_ae].item() - l_ae_v.v[0, node_ae].item()
                    # + NUDGE?

                # off-diagonal, same layer
                # l_v.one_spike is always true in this example -> ensure just one node in this layer spikes
                for other_ae_node in range((node_ae + 1), l_ae_v.n):
                    # node_column is not in refractory period either
                    if l_ae_v.refrac_count[0, other_ae_node].item() == 0:
                        other_nr = other_ae_node + encoding['Ae']
                        # penalty to make sure nodes do not spike at the same time
                        qubo[(nr_ae, other_nr)] = penalties_and_rewards['Ae']

                # off-diagonal, Inputs from layer X (connection X->Ae)
                # we work in the upper triangular matrix, connection goes from row to column
                for node_x in range(self.layers['X'].n):
                    inp = s_view_x[node_x] * conn[('X', 'Ae')].w[node_x, node_ae].item()
                    nr_x = node_x + encoding['X']
                    qubo[(nr_x, nr_ae)] = -1 * inp
                    inputs['Ae'][0, node_ae] += inp

                # off-diagonal, Inputs from layer Ai (connection Ai->Ae)
                # we work in the upper triangular matrix, connection goes from column to row
                # actual connections (where weights ≠ 0) are only where node_ae ≠ node_ai
                for node_ai in range(self.layers['Ai'].n):
                    if not node_ae == node_ai:
                        inp = s_view_ai[node_ai] * conn[('Ai', 'Ae')].w[node_ai, node_ae].item()
                        column_nr = node_ai + encoding['Ai']
                        qubo[(nr_ae, column_nr)] = -1 * inp
                        inputs['Ae'][0, node_ae] += inp

            # else: reward can be omitted for excitatory neurons -> done here for performance reasons

        # Layer Ai of inhibitory LIF-Nodes
        # needed later for Inputs from layer Ae (connection Ae->Ai)
        s_view_ae = self.layers['Ae'].s.float().view(self.layers['Ae'].s.size(0), -1)[0].tolist()

        for node in range(l_ai_v.n):
            # Could spike -> needs constraints
            if l_ai_v.refrac_count[0, node].item() == 0:
                nr_ai = node + encoding['Ai']

                # diagonal
                # = threshold - (voltage + connection-bias)
                # for c in conn: not needed in this case: always bias = 0
                    # if c[1] == 'Ai':
                        # l_ai_v.v[0, node] += conn[c].b[node]
                qubo[(nr_ai, nr_ai)] = l_ai_v.thresh.item() - l_ai_v.v[0, node].item() # + NUDGE?

                # for off-diagonal, same layer: nothing to do

                # off-diagonal, Inputs from layer Ae (connection Ae->Ai)
                # we work in the upper triangular matrix, connection goes from row to column
                # actual connections (where weights ≠ 0) are only where node_ai = node_ae
                inp = s_view_ae[node] * conn[('Ae', 'Ai')].w[node, node].item()
                nr_ae = node + encoding['Ae']
                qubo[(nr_ae, nr_ai)] = -1 * inp
                inputs['Ai'][0, node] += inp
            else: # might just have spiked -> needs reward to clamp qubit to 1
                if s_view_ai[node]:  # reward would not harm if neuron did not spike, but embedding easier without
                    nr_ai = node + encoding['Ai']
                    qubo[(nr_ai, nr_ai)] = penalties_and_rewards['Ai']


        # Wegen Hin- und Rückconnection: Verfälscht "Rüberklappen" Inhalt,
        # da connection von Ae nach Ai ≠ connection von Ai nach Ae?
        # -> Nein, denn: Wenn connection Ae->Ai existiert, dann existiert nicht wirklich Ai->Ae und umgekehrt
        # Für Situationen, wo dies doch der Fall wäre (Recurrent NN), wäre folgende Argumentation möglich,
        # solange nicht geclamped wird (da bei clamping Ae-Neuron spikt,
        # obwohl nicht in refrac-Period (in aktueller Implementierung), hält dann 1. nicht):
        # 1. Wenn Input-spike auf connection Ae->Ai, dann Ae-neuron in refrac-Period
        # -> bekommt keine Input-spikes Ai->Ae, sondern leer
        # -> ok, weil insgesamt bei Klappen der Wert dann nur der von Ae->Ai ist
        # 2. Wenn Input-spike auf connection Ai->Ae, dann Ai-neuron in refrac-Period
        # -> bekommt keine Input-spikes Ae->Ai, sondern leer
        # -> ok, weil insgesamt bei Klappen der Wert dann nur der von Ai->Ae ist
        # 3. Beide gleichzeitig Input-Spike -> beide in refrac-Period -> auch kein Problem: Wert leer
        # 4. Beide gleichzeitig kein Input-Spike -> Wert auf beiden Connections = w * 0 = 0
        # => "Rüberklappen" kein Problem, da keine Verfälschung

        # call Quantum Annealer or simulator (creates a triangular matrix out of qubo by itsself)
        # start = clock.time()
        # originally num_repeats=40, seems to work well with num_repeats=1, too (-> now default)
        solution = qbs.QBSolv().sample_qubo(qubo, num_repeats=num_repeats, verbosity=-1)
        # end = clock.time()
        # elapsed = end - start
        # print("\n Wall clock time qbsolv: %fs" % elapsed)
        print("\n Energy of qbsolv-solution: %f" % solution.first.energy)

        for l in self.layers:
            l_v = self.layers[l]
            # write spikes from (first) solution by filtering out 1s from neurons in refractory period
            if not l == 'X':  # not layer of Input-Nodes
                spikes = torch.full((1, l_v.n), False, dtype=torch.bool)
                nr = encoding[l]
                for node in range(l_v.n):
                    # is not in refractory period (has not just spiked) -> could spike
                    if l_v.refrac_count[0, node].item() == 0:
                        if solution.first.sample[nr + node] == 1:
                                spikes[0][node] = True
                l_v.s = spikes

                # Integrate inputs into voltage
                # (bias not necessary, as always = 0 in this example, inputs are zero where in refrac-period
                # -> no need to check, Input-layer does not have voltage)
                l_v.v += inputs[l]

                # Decrement refractory counters. (Input-layer does not have refrac_count)
                l_v.refrac_count = (l_v.refrac_count > 0).float() * (l_v.refrac_count - l_v.dt)

                # Refractoriness, voltage reset, and adaptive thresholds.
                l_v.refrac_count.masked_fill_(l_v.s, l_v.refrac)
                l_v.v.masked_fill_(l_v.s, l_v.reset)
                if l == 'Ae':  # layer of DiehlAndCookNodes
                    l_v.theta += l_v.theta_plus * l_v.s.float().sum(0)

                # from super().forward(...) (-> already called for Input-nodes)
                # -> l_v.x not used apart from in Input layer
                #if l_v.traces:
                    # Decay and set spike traces.
                    #l_v.x *= l_v.trace_decay
                    # Since l_v.traces_additive is always false
                    #l_v.x.masked_fill_(l_v.s != 0, 1)
    # end of forward_qa


    def run(
        self, inputs: Dict[str, torch.Tensor], time: int, num_repeats: int, one_step=False, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time. Adjusted for using QA

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.
        :param int num_repeats: Number of iterations the QA-simulator runs the problem

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet_qa.network import Network
            from bindsnet_qa.network.nodes import Input
            from bindsnet_qa.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        # unclamps = kwargs.get("unclamp", {}) -> not used
        masks = kwargs.get("masks", {})
        # injects_v = kwargs.get("injects_v", {}) -> not used

        # Compute reward. -> not used
        # if self.reward_fn is not None:
            # kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # Dynamic setting of batch size. -> not used
        # if inputs != {}:
            # for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                # if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    # inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                # elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    # inputs[key] = inputs[key].unsqueeze(1)

            # for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                # if inputs[key].size(1) != self.batch_size:
                    # self.batch_size = inputs[key].size(1)

                    # for l in self.layers:
                        # self.layers[l].set_batch_size(self.batch_size)

                    # for m in self.monitors:
                        # self.monitors[m].reset_state_variables()

                # break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # calculate possible Quantum Annealing penalties once
        penalties_and_rewards = {}
        # for l in self.layers:
        # if isinstance(self.layers[l], DiehlAndCookNodes): -> only layer Ae
        # if self.layers[l].one_spike: -> always the case
        penalties_and_rewards['Ae'] = self.penalty_one_spike(layer='Ae')
        #for all inhibitory layers (can be omitted for excitatory layers for performance reasons)
        # -> here only layer Ai
        penalties_and_rewards['Ai'] = self.reward_inhibitory(layer='Ai')


        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):

            # for l in inputs: -> only one, namely X
            # compute spikes of Input-layer X right away
            self.layers['X'].forward(x=inputs['X'][t])

            # forward-step with quantum annealing
            # start = clock.time()
            self.forward_qa(penalties_and_rewards, num_repeats=num_repeats)
            # end = clock.time()
            # elapsed = end - start
            # print("\n Wall clock time forward_qa(): %fs" % elapsed)

            # for l in self.layers: -> happens just for layer Ae

            # Clamp neurons to spike. -> happens just for layer Ae
            clamp = clamps.get('Ae', None)
            # if clamp is not None: -> always the case with this layer
            # if clamp.ndimension() == 1: -> always the case with this layer
            self.layers['Ae'].s[:, clamp] = 1
            # else: -> not used
                # self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.-> not used -> kick out?
                # unclamp = unclamps.get(l, None)
                # if unclamp is not None:
                    # if unclamp.ndimension() == 1:
                        # self.layers[l].s[unclamp] = 0
                    # else:
                        # self.layers[l].s[unclamp[t]] = 0

                # Inject voltage to neurons. -> not used -> kick out?
                # inject_v = injects_v.get(l, None)
                # if inject_v is not None:
                    # if inject_v.ndimension() == 1:
                        # self.layers[l].v += inject_v
                    # else:
                        # self.layers[l].v += inject_v[t]

            # Run synapse updates.
            for c in self.connections:
                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Module":
        # language=rst
        """
        Sets the node in training mode.

        :param mode: Turn training on or off.

        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)
