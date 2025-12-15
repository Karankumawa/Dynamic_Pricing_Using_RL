# Dynamic Pricing Using Reinforcement Learning (RL)

## 🚀 Project Overview

This project implements a **Dynamic Pricing System for E-Commerce** using **Reinforcement Learning (Q-Learning and Deep Q-Learning)**. The system intelligently adjusts product prices in real time based on market conditions, demand, and historical performance to maximize revenue.

A live demo of the project is deployed using **Streamlit**.

🔗 **Live App:** [https://dynamic-pricing-rl-01.streamlit.app/](https://dynamic-pricing-rl-01.streamlit.app/)

🔗 **GitHub Repository:** [https://github.com/Karankumawa/Dynamic_Pricing_Using_RL](https://github.com/Karankumawa/Dynamic_Pricing_Using_RL)

---

## 🎯 Key Features

* Dynamic price optimization using Reinforcement Learning
* Q-Learning and Deep Q-Learning implementations
* Real-time simulation of pricing strategies
* Interactive Streamlit dashboard
* Visual insights into rewards, pricing actions, and performance
* Modular and scalable codebase

---

## 🧠 Technologies Used

* **Python 3.9+**
* **Reinforcement Learning (Q-Learning, DQN)**
* **TensorFlow / Keras**
* **NumPy & Pandas**
* **Matplotlib**
* **Streamlit**

---

## 📂 Project Structure

```
Dynamic_Pricing_Using_RL/
│── app.py                     # Streamlit application
│── environment.py             # Pricing environment
│── q_learning_agent.py        # Q-learning implementation
│── dqn_agent.py               # Deep Q-learning implementation
│── utils.py                   # Helper functions
│── requirements.txt           # Project dependencies
│── README.md                  # Project documentation
│── models/                    # Saved models
│── assets/                    # Images and plots
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Karankumawa/Dynamic_Pricing_Using_RL.git
cd Dynamic_Pricing_Using_RL
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

* **Windows:**

```bash
venv\Scripts\activate
```

* **Mac/Linux:**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

```bash
streamlit run app.py
```

The app will open automatically in your browser at:

```
http://localhost:8501
```

---

## 🌐 Deployment on Streamlit Cloud

1. Push your project to **GitHub**
2. Go to **[https://streamlit.io/cloud](https://streamlit.io/cloud)**
3. Click **New App**
4. Select:

   * Repository: `Dynamic_Pricing_Using_RL`
   * Branch: `main`
   * Main file path: `app.py`
5. Click **Deploy** 🚀

---

## 📊 How It Works

* The environment simulates customer demand
* The RL agent selects price actions
* Rewards are calculated based on revenue
* The agent learns optimal pricing strategies over episodes

---

## 🧪 Future Enhancements

* Multi-product pricing
* Advanced RL algorithms (PPO, A2C)
* Market competitor simulation

---

## 👨‍💻 Author

**Karan Kumawat**
B.Tech Project – Reinforcement Learning

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ *If you like this project, don’t forget to star the repository!*
