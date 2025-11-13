import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Import from our other project files
from ecommerce_env import ECommerceEnv
from agent_utils import train_model, evaluate_policy, evaluate_baseline

# --- Streamlit Helper ---

@st.cache_resource
def get_env(config):
    """Cached environment creation."""
    env = ECommerceEnv(config)
    return env

# --- Main Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="RL Dynamic Pricing")
    st.title("🛒 Real-Time E-Commerce Dynamic Pricing Using Reinforcement Learning")
    
    # Use session state to store model
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None

    # Sidebar for Environment Configuration
    st.sidebar.header("1. Environment Configuration")
    st.sidebar.markdown("Define the parameters of your e-commerce simulation.")
    
    config = {}
    config['max_inventory'] = st.sidebar.number_input("Initial Inventory", 50, 1000, 100)
    config['max_timesteps'] = st.sidebar.number_input("Episode Length (Days)", 10, 90, 30)
    config['visitors_per_day'] = st.sidebar.number_input("Visitors per Day", 10, 200, 50)
    config['price_midpoint'] = st.sidebar.slider("Demand Price Midpoint ($)", 10, 40, 25,
        help="The price at which purchase probability is 50%.")
    config['demand_steepness'] = st.sidebar.slider("Demand Price Sensitivity", 0.05, 1.0, 0.2,
        help="How quickly demand drops as price increases (higher = more sensitive).")

    # Create the tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Project Overview", 
        "📈 Environment Simulation", 
        "🧠 Model Training", 
        "📊 Performance Evaluation"
    ])

    # --- Tab 1: Project Overview ---
    with tab1:
        st.header("Project Plan")
        st.markdown("""
        (This is the project plan you provided, loaded into the app.)
        
        ### 1. Objective of the Project
        The primary objective of this project is to design and develop a proof-of-concept system for dynamic pricing in an e-commerce environment using Reinforcement Learning (RL). The system will create an intelligent agent capable of making real-time pricing decisions to maximize a key business metric, such as revenue or profit.

        ### 2. Technology Stack
        - **Programming Language:** Python 3.9+
        - **Reinforcement Learning Frameworks:** Gymnasium (formerly OpenAI Gym) & Stable-Baselines3
        - **Numerical Computation & Data Handling:** NumPy & Pandas
        - **Data Visualization:** Matplotlib & Seaborn
        - **Web Interface:** Streamlit (this app)

        ### 3. Weekly Project Plan
        - **Week 1: Foundational Research and Environment Scoping**
            - Define State Space (inventory, time), Action Space (price points), and Reward (revenue).
            - Set up the development environment.
        - **Week 2: E-Commerce Simulation Environment Development**
            - Implement the custom `ECommerceEnv` class (see `ecommerce_env.py`).
            - Code the market simulation logic (the demand model).
        - **Week 3: RL Agent Implementation and Training**
            - Select the PPO algorithm from Stable-Baselines3 (see `agent_utils.py`).
            - Integrate the agent with the custom environment.
            - Train the agent (see 'Model Training' tab).
        - **Week 4: Performance Evaluation and Final Reporting**
            - Develop scripts to evaluate the trained agent against baselines (see `agent_utils.py`).
            - Generate visualizations to demonstrate learning (see 'Performance Evaluation' tab).
            - Compile the final project report.
        """)

    # --- Tab 2: Environment Simulation ---
    with tab2:
        st.header("Visualizing the Demand Curve")
        st.markdown("This chart shows the *probability* a single customer will purchase at a given price, based on the parameters you set in the sidebar.")

        env = get_env(config)
        prices = env.price_points
        probs = [env._demand_model(p) for p in prices]
        
        demand_df = pd.DataFrame({
            "Price ($)": prices,
            "Purchase Probability": probs
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Price ($)", y="Purchase Probability", data=demand_df, ax=ax, palette="viridis")
        ax.set_title("Customer Purchase Probability vs. Price")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        
        st.subheader("Environment Source Code")
        with st.expander("Click to see the ECommerceEnv class definition (from ecommerce_env.py)"):
            # Read the content of the environment file to display it
            try:
                with io.open("ecommerce_env.py", "r", encoding="utf-8") as f:
                    st.code(f.read())
            except FileNotFoundError:
                st.error("ecommerce_env.py not found. Make sure it's in the same directory as app.py.")

    # --- Tab 3: Model Training ---
    with tab3:
        st.header("Train the RL Agent (PPO)")
        st.markdown("This will train a Proximal Policy Optimization (PPO) agent on the custom environment. The trained model will be saved in this session for evaluation.")
        
        total_timesteps = st.number_input("Total Timesteps for Training", 10000, 1000000, 30000, 1000)
        
        if st.button("🚀 Start Training", type="primary"):
            st.info("Training started... this may take a few moments.")
            
            # Placeholders for the callback
            progress_bar = st.progress(0)
            text_area = st.empty()
            
            # Get the environment
            train_env = get_env(config)
            
            # Train the model (this function is now imported from agent_utils.py)
            model = train_model(train_env, total_timesteps, progress_bar, text_area)
            
            st.session_state.trained_model = model
            st.success("✅ Model training complete! You can now evaluate it in the next tab.")
            st.balloons()
        else:
            st.info("Click the button to begin training.")

    # --- Tab 4: Performance Evaluation ---
    with tab4:
        st.header("Compare Agent vs. Baselines")
        
        if st.session_state.trained_model is None:
            st.warning("Please train a model in the 'Model Training' tab first.")
        else:
            model = st.session_state.trained_model
            
            n_eval_episodes = st.slider("Number of Episodes to Evaluate", 10, 500, 100)
            
            if st.button("📊 Run Evaluation"):
                with st.spinner("Running evaluations..."):
                    eval_env = get_env(config)
                    
                    # Evaluate all policies (functions imported from agent_utils.py)
                    rl_revenues = evaluate_policy(model, eval_env, n_eval_episodes)
                    fixed_revenues = evaluate_baseline(eval_env, 'fixed_mid', n_eval_episodes)
                    random_revenues = evaluate_baseline(eval_env, 'random', n_eval_episodes)
                    
                    # --- Display Metrics ---
                    st.subheader("Average Total Revenue per Episode")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Trained RL Agent", f"${np.mean(rl_revenues):.2f}")
                    col2.metric("Fixed Price (Baseline)", f"${np.mean(fixed_revenues):.2f}")
                    col3.metric("Random Price (Baseline)", f"${np.mean(random_revenues):.2f}")

                    # --- Display Visualization ---
                    st.subheader("Distribution of Total Revenue")
                    
                    df = pd.DataFrame({
                        'RL Agent': rl_revenues,
                        'Fixed Price': fixed_revenues,
                        'Random Price': random_revenues
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df, ax=ax, palette="Set2")
                    ax.set_title("Comparison of Revenue Generation Strategies")
                    ax.set_ylabel("Total Revenue ($)")
                    st.pyplot(fig)
                    
                    # --- Show Policy ---
                    st.subheader("Learned Pricing Policy")
                    st.markdown("This table shows the price the *trained agent* chooses at different states.")
                    
                    inventory_levels = [config['max_inventory'], config['max_inventory'] * 0.5, config['max_inventory'] * 0.1]
                    timesteps = [0, config['max_timesteps'] * 0.5, config['max_timesteps'] - 1]
                    
                    policy_data = []
                    for inv in inventory_levels:
                        for t in timesteps:
                            obs = np.array([inv, t], dtype=np.float32)
                            action, _ = model.predict(obs, deterministic=True)
                            price = eval_env.price_points[action]
                            policy_data.append([f"{int(inv)} items", f"Day {int(t)}", f"${price}"])
                    
                    policy_df = pd.DataFrame(policy_data, columns=['Inventory', 'Time', 'Chosen Price'])
                    policy_pivot = policy_df.pivot(index='Inventory', columns='Time', values='Chosen Price').fillna("N/A")
                    st.dataframe(policy_pivot)
                    
                    st.markdown("""
                    **How to read this:**
                    - The agent may learn to set **higher prices** at the beginning (Day 0) when inventory is full.
                    - It might set **lower prices** near the end (e.g., Day 29) to sell off remaining stock.
                    - This demonstrates its ability to learn a *dynamic* policy, not just a fixed price.
                    """)


if __name__ == "__main__":
    # Run the Streamlit app
    main()