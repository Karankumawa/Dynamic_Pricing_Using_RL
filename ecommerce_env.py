import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- Week 2: E-Commerce Simulation Environment ---

class ECommerceEnv(gym.Env):
    """
    Custom OpenAI Gym Environment for E-Commerce Dynamic Pricing.
    
    Matches your project plan's description.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config=None):
        super(ECommerceEnv, self).__init__()
        
        self.config = config or {}
        
        # Define Price Points (Action Space)
        self.price_points = np.array([10, 15, 20, 25, 30, 35, 40])
        self.action_space = spaces.Discrete(len(self.price_points))
        
        # Define State Space: [current_inventory, time_step]
        self.max_inventory = self.config.get('max_inventory', 100)
        self.max_timesteps = self.config.get('max_timesteps', 30)  # e.g., 30 days
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.max_inventory, self.max_timesteps], dtype=np.float32),
            dtype=np.float32
        )
        
        # Demand curve parameters
        self.price_midpoint = self.config.get('price_midpoint', 25)
        self.demand_steepness = self.config.get('demand_steepness', 0.2)
        self.visitors_per_day = self.config.get('visitors_per_day', 50)
        
        # Initialize state
        self.state = None
        self.current_step = 0

    def _demand_model(self, price):
        """
        Calculates the probability of purchase based on price using a logistic function.
        More expensive -> lower probability.
        """
        # Sigmoid function: prob = 1 / (1 + exp(k * (price - mid)))
        prob = 1 / (1 + np.exp(self.demand_steepness * (price - self.price_midpoint)))
        return np.clip(prob, 0, 1)

    def step(self, action_idx):
        """
        Execute one time step within the environment.
        """
        # 1. Get the price from the action
        price = self.price_points[action_idx]
        
        # 2. Get current state
        current_inventory, time_step = self.state
        
        # 3. Simulate market logic
        purchase_prob = self._demand_model(price)
        
        # Simulate individual visitor decisions
        potential_sales = np.random.binomial(n=self.visitors_per_day, p=purchase_prob)
        
        # Actual sales are limited by current inventory
        sales = min(potential_sales, current_inventory)
        
        # 4. Calculate reward (revenue)
        revenue = sales * price
        reward = float(revenue) # Ensure reward is a float
        
        # 5. Update state
        new_inventory = current_inventory - sales
        new_time_step = time_step + 1
        self.state = np.array([new_inventory, new_time_step], dtype=np.float32)
        
        # 6. Check for termination
        terminated = (new_inventory <= 0) or (new_time_step >= self.max_timesteps)
        truncated = False  # We don't have a separate truncation condition
        
        # 7. info dict
        info = {
            'price': price,
            'sales': sales,
            'revenue': revenue,
            'purchase_prob': purchase_prob
        }
        
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment to an initial state.
        """
        super().reset(seed=seed)
        
        # Initial state: full inventory, time step 0
        self.state = np.array([self.max_inventory, 0], dtype=np.float32)
        self.current_step = 0
        
        info = {}
        return self.state, info

    def render(self):
        # In a real scenario, this might pop up a window.
        # For Streamlit, we'll render externally.
        pass

# --- Self-Test Block ---
if __name__ == "__main__":
    """
    If you run `python ecommerce_env.py`, this will check if the environment is valid.
    """
    from stable_baselines3.common.env_checker import check_env
    try:
        env = ECommerceEnv()
        check_env(env)
        print("ECommerceEnv check passed!")
        
        # Test a reset
        obs, _ = env.reset()
        print(f"Initial observation: {obs}")
        
        # Test a step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action} (Price: ${info['price']})")
        print(f"New Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        
    except Exception as e:
        print(f"ECommerceEnv check failed: {e}")