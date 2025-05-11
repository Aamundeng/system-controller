from jax.nn import relu, tanh, sigmoid
import jax.random as randjax

'''PIVOT CONFIGURATION PARAMETERS:'''

'''
________________________
0. RUN PARAMETERS
________________________
'''

# plant = 'Bathtub'
# plant = 'Cournot'
plant = 'Fishing'

# controller = 'PID'
controller = 'NN'

#target value of plant
target = 1.0
# Select whether to visualize plant value
plot_plant = False

'''
____________________________
1. CONTROL-SYSTEM PARAMETERS
____________________________
'''

epochs = 501
timesteps = 10
learning_rate_pid = 0.05 # Recommended for PID
learning_rate_nn = 0.005 # Recommended for NN
warmup_time = 0.2 # proportion of epochs used to warmstart learning rate
cooldown_time = 0.2 # proportion of epochs used to decay learning rate at the end
noise_range = 0.2
seed = 0
key = randjax.PRNGKey(seed)

'''
____________________________
2. CONTROLLER PARAMETERS:
____________________________
'''

kp_init = 0.1  # Proportional gain
ki_init = 0.05  # Integral gain
kd_init = 0.01  # Derivative gain

hidden_layer_sizes = [5, 5, 5]
init_range_nn = 0.1 # Initial range for weights

# activation_func = relu
# activation_func = tanh
activation_func = sigmoid

'''
____________________________
3. PLANT PARAMETERS:
____________________________
'''

#Bathtub:
# Cross-sectional area of the bathtub
A = 5.0
# Cross-sectional area of the drain
C = A / 50
# Initial height of the water
H_init = target
# Gravitational constant
g = 9.81


# Cournot competition:
# Maximum price
p_max = 3.0
# Marginal cost
c_m = 0.01
# Initial production quantities
q_1_init = 0.5
q_2_init = 0.5

# Fishery:
# Initial fish population
F_init = target
# Intrinsic growth rate
r = 0.2
# Maximum sustainable fish population
N = 3.0
