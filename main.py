import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as randjax
import optax
from flax import linen as nn
from config import *


'''
________________________________________________________  
MODULE 1: Running the system and plotting the results  
________________________________________________________ 
'''


def run_system_and_plot_results(plant_name, controller_name, plot_plant=False):
    match plant_name:
        case 'Bathtub':
            plant = Bathtub(H_init, A, C, g)
        case 'Cournot':
            plant = Cournot(p_max=p_max, c_m=c_m, q_1_init=q_1_init, q_2_init=q_2_init)
        case 'Fishing':
            plant = Fishing(F_init=F_init, r=r, N=N)
    match controller_name:
        case 'PID':
            controller = PID_Controller(kp=kp_init, ki=ki_init, kd=kd_init, lr=learning_rate_pid,
                                        epochs=epochs, warmup_time=warmup_time, cooldown_time=cooldown_time)
        case 'NN':
            controller = NN_Controller(hidden_layer_sizes=hidden_layer_sizes, activation_func=activation_func,
                                       init_range=init_range_nn, lr=learning_rate_nn, key=key,
                                       epochs=epochs, warmup_time=warmup_time, cooldown_time=cooldown_time)

    consys = ConSys(plant=plant, controller=controller, target=target, epochs=epochs, timesteps=timesteps,
                    noise_range=noise_range, key=key)
    results, visualize_results = consys.run_system()
    print("Training done:")
    # 1. The configuration parameters for the run, presented in a table:
    if isinstance(controller, PID_Controller):  # PID: Plot results, which is a list of (mse,kp,ki,kd) per epoch
        mses, kps, kis, kds = zip(*results)
        print(f"Final loss:{mses[-1]}")
        print(f"Parameters:")
        print(f"Last kp: {kps[-1]}, Last ki: {kis[-1]}, Last kd: {kds[-1]}")
        plt.plot(mses)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE, ' + 'Plant: ' + plant_name + ' Controller: ' + controller_name)
        plt.show()

        plt.plot(kps, label='kp')
        plt.plot(kis, label='ki')
        plt.plot(kds, label='kd')
        plt.title('PID values, ' + 'Plant: ' + plant_name + ' Controller: ' + controller_name)
        plt.xlabel('Epoch')
        plt.ylabel('PID values')
        plt.legend()
        plt.show()

    else:  # Neural Network: Plot results, which is a list of mse's per epoch
        print(f"Final loss:{results[-1]}")
        plt.plot(results)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE, ' + 'Plant: ' + plant_name + ' Controller: ' + controller_name)
        plt.show()


    # Plot plant_value in the chosen epochs
    if plot_plant:
        for epoch, plant_values in visualize_results:
            plt.plot(plant_values)
            plt.xlabel('Timestep')
            plt.ylabel('Plant value')
            plt.title('Epoch: ' + str(epoch) + ' Plant: ' + plant_name + ' Controller: ' + controller_name)
            plt.show()



'''
________________________________________________________  
MODULE 2: Defining the plants and their separate "simulation functions"
________________________________________________________  
'''

class Plant:
    '''
    Responsibilities of the plant:
        - give the plant-value to consys for error calculation (plant_value)
        - receive control signal from consys (which has received it from the controller) and update the plant accordingly (update_plant)
        - reset the plant to its initial state at each new epoch (reset_plant)
    '''
    
    def __init__(self, vars_init,  constants, sim_func):
        self.vars_init = vars_init # initial variable values of plant
        self.vars = vars_init # current variable values of plant
        self.constants = constants # package of plant-specific constants
        self.sim_func = sim_func # function to update or give value for the chosen plant
        
    def plant_value(self):
        value = self.sim_func(self.vars, self.constants, update=False)
        return value

    def update_plant(self, U, D):
        self.vars = self.sim_func(self.vars, self.constants, U, D, update=True)
        
    def reset_plant(self):
        self.vars = self.vars_init

class Bathtub(Plant):
    def __init__(self, H_init, A, C, g=9.81):
        super().__init__(H_init, (A,C,g), bathtub_sim)
    
class Cournot(Plant):
    def __init__(self, q_1_init, q_2_init, p_max, c_m):
        super().__init__((q_1_init, q_2_init), (p_max, c_m), cournot_sim)

class Fishing(Plant):
    def __init__(self, F_init, r, N):
        super().__init__(F_init, (r, N), fishing_sim)

#___________________________

def bathtub_sim(vars, constants, U=0, D=0, update=False):
    H = vars
    A, C, g = constants
    if update:
        V = jnp.sqrt(2 * g * H) # velocity
        Q = V * C # flow rate
        B = H * A # current volume
        B = B - Q + U + D  # new volume
        H = B / A # new height
        vars = H
        return vars
    else: 
        val = H
        return val

def cournot_sim(vars, constants, U=0, D=0, update=False):
    q_1, q_2 = vars
    p_max, c_m = constants
    if update:
        q_1 = jnp.clip(q_1 + U, 0, 1)
        q_2 = jnp.clip(q_2 + D, 0, 1)
        vars = q_1, q_2
        return vars
    else:
        q = q_1 + q_2
        price = p_max - q
        val = q_1 * (price - c_m)
        return val
    
def fishing_sim(vars, constants, U=0, D=0, update=False):
    F = vars
    r, N, = constants
    if update:
        F_delta = r * F * (1 - F/N) + U + D
        F = jnp.maximum(0.001, F + F_delta) 
        vars = F
        return vars
    else: 
        val = F
        return val


'''
________________________________________________________
MODULE 3: Defining the control system
________________________________________________________
'''


class ConSys:
    '''
    Responsibilities of the control-system:
     - for each timestep in an epoch, take the plant-value and compute the error w.r.t the target-value. Give that error to the controller.
        receive the controllers signal and give that back to the plant. (timestep_iteration)
    - for an epoch, take the MSE of the target-plant errors from all timesteps  (mse_func)
    - Iterate through all epochs. for each epoch, compute the gradient of the MSE w.r.t the parameters. Give that gradient to the controller, and let
     the controller update its internal parameters. Also, store the epochs MSE and parameters for plotting. (run_system)

    Utility functions:
    - For certain epochs, store the plant-values for plotting. (visualize_consys)
    '''

    def __init__(self, plant, controller, target, epochs, timesteps, noise_range, key):
        self.plant = plant
        self.controller = controller
        self.target = target
        self.epochs = epochs
        self.timesteps = timesteps
        self.noise_range = noise_range
        self.key = key

        # Jit-compiled functions
        self.mse_func_jit = jax.jit(self.mse_func)  # loss
        self.grads_func_jit = jax.jit(jax.grad(self.mse_func))  # gradients

    def timestep_iteration(self, params, subkey, errors):
        plant_value = self.plant.plant_value()
        errors.append(self.target - plant_value)
        U = self.controller.compute_control_signal(params, errors)  # controller receive error and gives control signal
        D = randjax.uniform(subkey, (), minval=-self.noise_range, maxval=self.noise_range)  # uniform noise in range
        # () indicates we want a scalar output and not a tensor
        self.plant.update_plant(U, D)  # change value of plants according to signals
        return errors

    def mse_func(self, params, key):
        errors = []  # store errors throughout epoch
        self.plant.reset_plant()  # reset plant-value for new epochs
        subkeys = randjax.split(key, self.timesteps)  # get 'timesteps' number of random seeds/keys
        for timestep in range(self.timesteps):
            # compute signal for timestep and return updated error list
            errors = self.timestep_iteration(params, subkeys[timestep], errors)

        mse = jnp.mean(jnp.square(jnp.array(errors)))  # square each error and take the mean
        return mse

    def visualize_consys(self, key):
        # variant of mse_func where we extract the plant value at each timestep for visualization
        params = self.controller.params
        self.plant.reset_plant()
        plant_values = []
        errors = []
        subkeys = randjax.split(key, self.timesteps)
        for timestep in range(self.timesteps):
            errors = self.timestep_iteration(params, subkeys[timestep], errors)
            plant_values.append(self.plant.plant_value())  # only difference: get plant value for timestep
        return plant_values

    def run_system(self):
        visualize_results = []
        results = []
        subkeys = randjax.split(self.key, self.epochs)
        for epoch in range(self.epochs):
            # 1. Training: compute gradient of loss function and pass them to controller for updating
            params = self.controller.params
            gradients = self.grads_func_jit(params, subkeys[epoch])
            self.controller.update_params(gradients)

            # 2. Reporting: Store MSE and parameters (if PID) for this epoch, for plotting.
            mse = self.mse_func_jit(params, subkeys[epoch])
            if isinstance(self.controller, NN_Controller):
                results.append(mse)
            else:
                kp, ki, kd = params
                results.append((mse, kp, ki, kd))

            # 3. Visualizing: plant value for the first, middle, and last epoch, when plot_plant = True:
            if epoch % int(self.epochs / 2) == 0:
                visualize_results.append((epoch, self.visualize_consys(subkeys[epoch])))
        return results, visualize_results


'''
----------------------------------------------------------
MODULE 4: Defining the controllers: PID and Neural Network, and their 
"models", which describes how parameters interact with pid-values to create output.
----------------------------------------------------------
'''


class Controller:
    """
    Responsibilities of the controller:
        - compute control signal U from PID-values taken from the errors (Compute_control_signal)
        - get gradients from loss function and thence update parameters with adam gradient descent (Update_params)
    Utility functions:
        - calculate PID-values from errors (calculate_pid_values)
        - schedule the learning rate for the adam optimizer (make_schedule)
    """

    def __init__(self, params, model, lr, epochs, warmup_time=0.2, cooldown_time=0.2):
        self.model = model  # model decides how
        self.params = params
        # configure the scheduler (hard-coded): warm start for the 20% first epochs, then decay for the 20% last epochs
        scheduler = self.make_schedule(lr=lr, epochs=epochs, warmup_time=warmup_time,
                                       cooldown_time=cooldown_time, decay_rate=0.01)
        # make the optimizer, consisting of a grad.descent algorithm + internal state which contains
        # information on the parameters, the learning rate, and momentum info etc.
        self.optimizer = optax.adam(scheduler)
        self.optimizer_state = self.optimizer.init(self.params)

    def compute_control_signal(self, params, errors):
        pid_values = self.calculate_pid_values(errors)
        U = self.model.apply(params, pid_values)
        return U

    def calculate_pid_values(self, errors):
        proportional = errors[-1]
        integral = sum(errors)
        derivative = errors[-1] - errors[-2] if len(errors) > 1 else 0
        return jnp.array([proportional, integral, derivative])

    def update_params(self, grads):
        # updates = recommended increments to each parameter. learning rate etc. already factored in.
        updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        self.params = optax.apply_updates(self.params, updates)

    def make_schedule(self, lr, epochs, warmup_time, cooldown_time, decay_rate):
        warmup_end = int(warmup_time * epochs)
        decay_start = int(epochs - cooldown_time * epochs)
        if decay_start < warmup_end:
            raise ValueError("warmup-portion and decay-portion must sum to less than 1.0.")
        scheduler = optax.join_schedules(
            schedules=[
                optax.linear_schedule(init_value=0, end_value=lr, transition_steps=warmup_end),  # warmup
                optax.constant_schedule(value=lr),  # constant lr until decay
                optax.exponential_decay(init_value=lr, transition_steps=epochs - decay_start, decay_rate=decay_rate)],
            # decay
            boundaries=[warmup_end, decay_start]
            # end warmup at warmup_end, then constant, then exponential decay at decay_start
        )
        return scheduler


class PID_Controller(Controller):
    def __init__(self, kp, ki, kd, lr, epochs, warmup_time=0.2, cooldown_time=0.2):
        model = PID_Model()
        params = jnp.array([kp, ki, kd])
        super().__init__(params, model, lr, epochs, warmup_time, cooldown_time) # let superclass configure the optimizer


class NN_Controller(Controller):
    def __init__(self, hidden_layer_sizes, activation_func, init_range, lr, key, epochs, warmup_time=0.2, cooldown_time=0.2):
        model = NN_Model(hidden_layer_sizes=hidden_layer_sizes, activation_func=activation_func,
                         init_range=init_range)
        params = model.init(key, jnp.ones((1, 3))) # initialize the network to accept a 1x3 tensor [P,I,D]
        super().__init__(params, model, lr, epochs, warmup_time, cooldown_time)

#_______________________

class PID_Model:
    def apply(self, params, pid_values):
        return jnp.dot(params, pid_values)


class NN_Model(nn.Module):
    hidden_layer_sizes: list
    activation_func: callable
    init_range: float

    @nn.compact
    def __call__(self, x):
        for hidden_layer_size in self.hidden_layer_sizes:  # make hidden layers
            x = nn.Dense(features=hidden_layer_size,
                         kernel_init=nn.initializers.uniform(scale=self.init_range),
                         bias_init=nn.initializers.uniform(scale=self.init_range / 2)
                         )(x)
            x = self.activation_func(x)
        x = nn.Dense(features=1,
                     kernel_init=nn.initializers.uniform(scale=self.init_range),
                     bias_init=nn.initializers.uniform(scale=self.init_range / 2)
                     )(x)  # make output layer
        return x[0] # return first value, which contains the number output of the neural net


'''
__________________________
RUN CODE:
__________________________
'''

def main():
    run_system_and_plot_results(plant, controller, plot_plant=plot_plant)

if __name__ == "__main__":
    main()




