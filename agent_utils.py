import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --- Helper Classes and Functions ---

class StreamlitCallback(BaseCallback):
    """
    A custom callback to update a Streamlit progress bar.
    """
    def __init__(self, progress_bar, text_area, total_timesteps, verbose=0):
        super(StreamlitCallback, self).__init__(verbose)
        self.progress_bar = progress_bar
        self.text_area = text_area
        self.total_timesteps = total_timesteps
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Update progress bar
        progress = self.num_timesteps / self.total_timesteps
        
        # --- THIS IS THE FIX ---
        # We "clamp" the value, ensuring it is never greater than 1.0
        # This prevents the StreamlitAPIException
        clamped_progress = min(progress, 1.0)
        
        # We use the new, safe clamped_progress value here
        self.progress_bar.progress(clamped_progress)
        # --- END OF FIX ---

        # Update text area
        elapsed_time = time.time() - self.start_time
        steps_per_second = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        remaining_timesteps = self.total_timesteps - self.num_timesteps
        eta = remaining_timesteps / steps_per_second if steps_per_second > 0 else 0
        
        self.text_area.text(
            f"Training... Timestep: {self.num_timesteps}/{self.total_timesteps}\n"
            f"Steps/sec: {steps_per_second:.2f}\n"
            f"Elapsed Time: {elapsed_time:.2f}s\n"
            f"ETA: {eta:.2f}s"
        )
        
        return True

# --- Week 3: RL Agent Training ---

def train_model(env, total_timesteps, progress_bar, text_area):
    """
    Instantiates, trains, and returns a PPO agent.
    """
    callback = StreamlitCallback(progress_bar, text_area, total_timesteps)
    
    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=0)
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    return model

# --- Week 4: Performance Evaluation ---

def evaluate_policy(model, env, n_episodes=100):
    """
    Evaluate a trained RL agent.
    """
    all_rewards = []
    all_revenues = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_revenue = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_revenue += info['revenue']
        all_rewards.append(total_reward)
        all_revenues.append(total_revenue)
    return all_revenues

def evaluate_baseline(env, strategy, n_episodes=100):
    """
    Evaluate a baseline strategy (fixed or random).
    """
    all_revenues = []
    
    if strategy == 'fixed_mid':
        action = len(env.price_points) // 2
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_revenue = 0
        while not done:
            if strategy == 'random':
                action = env.action_space.sample()
            else: # fixed_mid
                action = action # Use the pre-defined mid-price action
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_revenue += info['revenue']
        all_revenues.append(total_revenue)
    return all_revenues