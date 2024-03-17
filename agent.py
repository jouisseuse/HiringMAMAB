from sklearn.linear_model import Ridge
import numpy as np

'''
code for M agent
    - estimate coefficient of arms
    - make bid choices
    - record the result
'''


class Agent:
    def __init__(self, agent_id, n_arms, group, round, feature_vector_length):
        self.id = agent_id
        self.n_arms = n_arms
        self.group = group
        self.round = round
        self.feature_vector_length = feature_vector_length
        self.estimate_coefficient = {} # key:group, value: (coefficient_mu, coefficient_sigma)
        self.rewards = np.zeros((round, 2)) # [(group_id, reward)]
        self.history_dict = {} # key:group, value:list of (feature_vector, reward)
        # Initialize strategy-specific parameters

        self.current_feature_vector = None
        self.current_arms_group = None

    def select_arm(self):
        pass

    def update(self, current_round, chosen_arm, reward):
        chosen_group = self.current_arms_group[chosen_arm]
        feature_vector = self.current_feature_vector[chosen_arm]
        self.rewards[current_round] = [chosen_group, reward]
        self.history_dict[chosen_group].append((feature_vector, reward))

    def get_current_feature_vector(self, feature_vector):
        self.current_feature_vector = feature_vector # [arm_num, feature_vector_length]

    def get_current_arms_group(self, current_arms_group):
        self.current_arms_group = current_arms_group # [arm_num,1]

    def get_all_reward(self):
        # calculate the total reward for all round
        return self.rewards[:,1].sum()

class ThompsonSamplingAgent(Agent):
    def __init__(self, agent_id, n_arms, group, round, feature_vector_length):
        super().__init__(agent_id, n_arms, group, round, feature_vector_length)
        # Initialize Thompson Sampling-specific parameters
        self.mu_0 = np.zeros(self.feature_vector_length) # Initialize the prior mean as a vector of zeros
        self.sigma_0 = np.eye(self.feature_vector_length) # Initialize the prior covariance as the identity matrix


    def select_arm(self):
        # Implement arm selection logic for Thompson Sampling
        # chose the ith arm with highest estimated reward
        expected_rewards = [self.get_estimate_reward(i) for i in range(self.n_arms)]
        expected_rewards = np.array(expected_rewards)
        return expected_rewards.argmax()

    def select_arm_group(self, chosen_group, feature_vector, reward):
        # initialize the group reward
        self.history_dict[chosen_group] = list()
        self.history_dict[chosen_group].append((feature_vector, reward))
        self.update_group_estimate(chosen_group)
          

    def get_estimate_reward(self, chosen_arm):
        # get ith arm's group
        group_id = self.current_arms_group[chosen_arm]
        beta_sampled = np.random.multivariate_normal(self.estimate_coefficient[group_id][0],self.estimate_coefficient[group_id][1])
        expected_reward = np.dot(beta_sampled, self.current_feature_vector[chosen_arm])
        # return the estimated reward for chosen arm
        return expected_reward

    def update_group_estimate(self, chosen_arm_group):
        # Update parameters based on observed reward
        group_history = self.history_dict[chosen_arm_group]
        feature_vector_array = np.array([itm[0] for itm in group_history])
        reward_array = np.array([itm[1] for itm in group_history])
        self.estimate_coefficient[chosen_arm_group] = self.update_coefficient_estimate(feature_vector_array, reward_array)

    # Thompson Sampling version
    # sigma_squared - Noise variance (Assuming known for simplicity)
    def update_coefficient_estimate(self, Xg, Yg, sigma_squared=1.0):
        # Calculating the posterior mean (mu_n) and covariance (Sigma_n) of beta
        Sigma_n_inv = np.linalg.inv(self.sigma_0) + (1 / sigma_squared) * Xg.T.dot(Xg)
        sigma_n  = np.linalg.inv(Sigma_n_inv)
        mu_n = sigma_n.dot(np.linalg.inv(self.sigma_0).dot(self.mu_0) + (1 / sigma_squared) * Xg.T.dot(Yg))
        
        return (mu_n, sigma_n)



class UCBAgent(Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # Initialize UCB-specific parameters

    def select_arm(self):
        # Implement arm selection logic for UCB
        pass

    def update(self, chosen_arm, reward):
        # Update parameters based on observed reward for UCB
        pass



############## code for test ####################
# Ridge Regression version
# def update_coefficient_estimate(Xg, Yg, alpha=1.0):
#     """
#     Update the estimate of the skill coefficient using Ridge Regression.

#     Parameters:
#     - Xg: The feature dataset, where each row represents the feature vector of a hired worker.
#     - Yg: The skill dataset, where each value corresponds to the skill value of a hired worker.
#     - alpha: The regularization coefficient for Ridge Regression.

#     Returns:
#     - theta_g_hat: The updated estimate of the skill coefficient.
#     """
#     # Initialize the Ridge Regression model
#     model = Ridge(alpha=alpha)
    
#     # Fit the Ridge Regression model using the provided data
#     model.fit(Xg, Yg)
    
#     # Retrieve the updated coefficients as the estimated skill values
#     theta_g_hat = model.coef_
    
#     return theta_g_hat
# Example data
# Xg = np.array([[1, 2, 5], [3, 4, 9], [5, 6, 7]])  # Example feature dataset
# Yg = np.array([3, 5, 7])  # Example skill dataset

# Update the skill coefficient estimate using the example data and default regularization coefficient
# theta_g_hat = update_coefficient_estimate(Xg, Yg)
# print("Updated estimate of the skill coefficient:", theta_g_hat)



