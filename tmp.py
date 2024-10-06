import numpy as np

def calculate_reward_probability_vector(arm_feature, agent_type_mix, theta):
    theta_type1 = np.array(theta[:4])  # First 4 elements of theta for Type 1
    theta_type2 = np.array(theta[4:])  # Last 4 elements of theta for Type 2
    reward = (
        agent_type_mix[0] * np.dot(arm_feature, theta_type1) +  # Contribution from Type 1
        agent_type_mix[1] * np.dot(arm_feature, theta_type2)    # Contribution from Type 2
    )
    return reward

# Global theta (new)
theta = [0.4, 0.4, 0.4, 0.4, # Type 1: [theta_A1_Type1, theta_A2_Type1, theta_B1_Type1, theta_B2_Type1]
         0.4, 0.4, 0.4, 0.4]   # Type 2: [theta_A1_Type2, theta_A2_Type2, theta_B1_Type2, theta_B2_Type2]

# Agent type mixes
agents = [
    [0.90, 0.10],  # Agent 1
    [0.70, 0.30],  # Agent 2
    [0.30, 0.70],  # Agent 3
    [0.10, 0.90]   # Agent 4
]

# Arm feature vectors (4-dimensional one-hot encoded)
arm_features = [
    [1, 0, 1, 0],  # Arm 1: A1, B1
    [0, 1, 1, 0],  # Arm 2: A2, B1
    [1, 0, 0, 1],  # Arm 3: A1, B2
    [0, 1, 0, 1]   # Arm 4: A2, B2
]

# # Calculate reward for each agent on each arm
# for i, agent in enumerate(agents):
#     print(f"Agent {i+1} type mix: {agent}")
#     for j, arm in enumerate(arm_features):
#         reward = calculate_reward_probability_vector(arm, agent, theta)
#         print(f"  Reward for Arm {j+1}: {reward:.2f}")
#     print()

print(np.random.normal(0, 0.1))