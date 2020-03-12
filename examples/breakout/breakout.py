from bindsnet_qa.network import Network
from bindsnet_qa.pipeline import EnvironmentPipeline
from bindsnet_qa.encoding import bernoulli
from bindsnet_qa.network.topology import Connection
from bindsnet_qa.environment import GymEnvironment
from bindsnet_qa.network.nodes import Input, LIFNodes
from bindsnet_qa.pipeline.action import select_softmax

# Build network.
network = Network(dt=1.0)

# Layers of neurons.
inpt = Input(n=80 * 80, shape=[80, 80], traces=True)
middle = LIFNodes(n=100, traces=True)
out = LIFNodes(n=4, refrac=0, traces=True)

# Connections between layers.
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(source=middle, target=out, wmin=0, wmax=1)

# Add all layers and connections to the network.
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Load the Breakout environment.
environment = GymEnvironment("BreakoutDeterministic-v4")
environment.reset()

# Build pipeline from specified components.
pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=bernoulli,
    action_function=select_softmax,
    output="Output Layer",
    time=100,
    history_length=1,
    delta=1,
    plot_interval=1,
    render_interval=1,
)

# Run environment simulation for 100 episodes.
for i in range(100):
    total_reward = 0
    pipeline.reset_state_variables()
    is_done = False
    while not is_done:
        result = pipeline.env_step()
        pipeline.step(result)

        reward = result[1]
        total_reward += reward

        is_done = result[2]
    print(f"Episode {i} total reward:{total_reward}")
