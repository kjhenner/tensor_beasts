Let's consider the following: I want to create a simulated environment in which multiple neural network entities can interact.

I want to be able to represent the local environment state as a tensor. As such, I want to use a grid to represent the environment. Each grid space can be occupied by a single entity.

There will be four different kinds of entity:

1. passive objects.
2. plants
3. herbivores
4. predators

All entities have observable state. This observable state is stored in the world tensor.

The state of the passive objects can be observed and acted on by 


get_rand: 0:00:00.000055
get_mask: 0:00:00.000478
get_growth_rand: 0:00:00.001571
get_growth_tensor: 0:00:00.000473
get_crowding: 0:00:00.007630
get_seed: 0:00:00.004474
update: 0:00:00.006154
display: 0:00:00.008794

get_rand: 0:00:00.000002
get_mask: 0:00:00.000449
get_growth_rand: 0:00:00.001684
get_growth_tensor: 0:00:00.000751
get_crowding: 0:00:00.007851
get_seed: 0:00:00.004944
update: 0:00:00.005740
display: 0:00:00.006468