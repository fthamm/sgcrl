import os, pickle
import numpy as np
import jax
import haiku as hk

from contrastive.utils import make_environment
from contrastive.networks import make_networks
from my_envs.stacked_blocks_env import StackedBlocksEnv


import tensorflow as tf
import haiku as hk
import re
import numpy as np
import jax.numpy as jnp

def load_params_for_inference(checkpoint_dir):
    """Load parameters from checkpoint directory with multiple fallback methods."""
    
    # Method 1: Try to find pickle file (easiest method)
    pickle_path = os.path.join(checkpoint_dir, 'learner_params.pkl')
    if os.path.exists(pickle_path):
        print(f"Loading from pickle file: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    # Method 2: Try to find TF checkpoint
    # Get the latest checkpoint file if a number wasn't specified
    if not checkpoint_dir.endswith('.index') and not os.path.exists(checkpoint_dir):
        checkpoints = tf.io.gfile.glob(os.path.join(checkpoint_dir, 'ckpt-*'))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            checkpoint_dir = latest
    
    # Remove .index extension if present
    if checkpoint_dir.endswith('.index'):
        checkpoint_dir = checkpoint_dir[:-6]
    
    print(f"Loading from TF checkpoint: {checkpoint_dir}")
    
    # Load the TF checkpoint
    reader = tf.train.load_checkpoint(checkpoint_dir)
    
    # Check if the Python state exists in the checkpoint
    if 'learner/.ATTRIBUTES/py_state' in reader.get_variable_to_shape_map():
        # Extract the Python state object
        py_state = reader.get_tensor('learner/.ATTRIBUTES/py_state')
        
        # Try to extract the policy parameters
        try:
            # The py_state might be a serialized pickle - try to unpack it
            import io
            try:
                state = pickle.loads(py_state)
                print("Successfully unpacked pickle from py_state")
                
                # State should be a TrainingState object
                if hasattr(state, 'policy_params'):
                    print("Found policy_params in TrainingState")
                    return state.policy_params
            except:
                print("Could not unpickle py_state, raw data returned")
                return py_state
        except:
            print("Could not extract policy parameters")
            return None
    
    print("Could not find learner state in checkpoint")
    return None

def main():
    checkpoint_dir = 'logs/contrastive_cpc_stacked_blocks_42/251c05c8-4839-11f0-864a-e1faa5c0879f/checkpoints/learner'

    # Try both with and without ckpt-13
    try:
        params = load_params_for_inference(checkpoint_dir)
        if params is None:
            params = load_params_for_inference(os.path.join(checkpoint_dir, 'ckpt-1'))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    
    # Print what we found
    if params is not None:
        if hasattr(params, 'keys'):
            print(f"Parameter keys: {params.keys()}")
        else:
            print(f"Parameter type: {type(params)}")


    # --- after loading `params` and printing them ---
    # 2) Build env + get obs_dim from checkpoint, not from env_spec
    env, _ = make_environment('stacked_blocks', 0, 0, seed=42)
    print(env.observation_spec())
    from acme import specs
    env_spec = specs.make_environment_spec(env)

    # infer obs_dim from the first policy‐network weight
    # policy net first linear layer is under 'mlp/~/linear_0'
    first_w = params['mlp/~/linear_0']['w']
    true_obs_dim = int(first_w.shape[0])
    print(f"Inferred obs_dim={true_obs_dim} from checkpoint")

    print(env_spec.observations.shape)
    print(true_obs_dim)

    # 3) Build exactly the same networks you trained with
    net_fn = make_networks
    networks = net_fn(spec=env_spec,
                      obs_dim=true_obs_dim,
                      repr_dim=256,
                      repr_norm=False,
                      twin_q=False,
                      use_image_obs=False,
                      hidden_layer_sizes=(256,256))

    @jax.jit
    def select_action(param_dict, obs_and_goal_jax):
        # obs_jax: jnp.ndarray with shape [obs_dim,]
        pi = networks.policy_network
        # add batch dim
        batched = obs_and_goal_jax[jnp.newaxis, :]
        # get distribution
        dist_params = pi.apply(param_dict, batched)
        # deterministic action
        action = networks.sample_eval(dist_params, None)
        # return shape [action_dim,]
        return action[0]

    for ep in range(3):
        goal = env.goal                  # shape (3,)
        timestep = env.reset()
        total_r = 0.0

        # initial 6‐dim obs+goal
        obs = timestep.observation       # shape (3,)
        full = np.concatenate([obs, goal], axis=-1)  # (6,)
        print("DEBUG initial full shape:", full.shape)   
        obs_and_goal_jax = jnp.array(full)            

        while True:
            # get action from JAX
            a_jax = select_action(params, obs_and_goal_jax)
            a = np.asarray(a_jax)       # shape (action_dim,)
            
            # step the environment
            timestep = env.step(a)
            total_r += timestep.reward          
            print(f"step → action={a}, reward={timestep.reward:.3f}")

            # build next obs+goal
            obs = timestep.observation
            full = np.concatenate([obs, goal], axis=-1)
            obs_and_goal_jax = jnp.array(full)

            # break on terminal
            if timestep.last():  
                break

        print(f"Episode {ep} total reward {total_r:.3f}\n")

if __name__=='__main__':
    main()