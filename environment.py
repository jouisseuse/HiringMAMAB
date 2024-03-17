from agent import Agent, ThompsonSamplingAgent
from bandit import ArmGroup, Arm

import numpy as np

class Environment:
    def __init__(self, n_arms, round, m_agents, g_groups, feature_vector_length):
        self.n_arms = n_arms
        self.round = round
        self.m_agents = m_agents
        self.g_groups = g_groups
        self.feature_vector_length =  feature_vector_length
        self.mu_x = 1.5 * np.ones(feature_vector_length)  # Mean of features
        self.sigma_x = 1  # Standard deviation of features
        self.sigma_epsilon = 0.5  # Standard deviation of noise

        ### object
        self.groups = self.set_up_groups()
        self.arms = None
        self.agents = self.set_up_agents()


    def set_up_groups(self):
        group_list = []
        for i in range(self.g_groups):
            group_coefficient = np.random.rand(self.feature_vector_length)
            group_list.append(ArmGroup(i, group_coefficient))

        return group_list
    
    def set_up_agents(self):
        agent_list = []
        for i in range(self.m_agents):
            agent_list.append(ThompsonSamplingAgent(i, self.n_arms, self.g_groups, self.round, self.feature_vector_length))
        
        return agent_list
    
    def set_up_arms(self):
        arm_list = []
        feature_vector = np.random.normal(self.mu_x, self.sigma_x, (self.n_arms, self.feature_vector_length))
        epsilon = np.random.normal(0, self.sigma_epsilon, self.n_arms)
        for i in range(self.n_arms):
            group_id = np.random.randint(self.g_groups)
            group_coefficients = self.groups[group_id].get_coefficients()
            new_arm = Arm(i, group_id, feature_vector[i], group_coefficients, epsilon[i])
            arm_list.append(new_arm)
        
        return arm_list
    
    def get_current_arms_group(self):
        # get current arms' group id from self.arms and return a list of group ids
        return [arm.get_group() for arm in self.arms]
    
    def get_current_feature_vector(self):
        # get current arms' feature vector from self.arms and return a list of feature vectors
        return [arm.get_feature_vector() for arm in self.arms]

    def initial_sample(self):
        # Initial sample, equal to the sum of group sizes
        for agent in self.agents:
            X_g = np.random.normal(self.mu_x, self.sigma_x, (self.g_groups, self.feature_vector_length))
            epsilon_g = np.random.normal(0, self.sigma_epsilon, self.g_groups)
            for gp in self.groups:
                Y_g = np.dot(X_g[gp.id], gp.get_coefficients()) + epsilon_g[gp.id]
                agent.select_arm_group(gp.id, X_g[gp.id], Y_g)
        
  
    
    def run_simulation(self):
        # Initial sample, equal to the sum of group sizes
        self.initial_sample()
        # Run the simulation
        for rd in range(self.round):
            print("========== Round: {} Start ==========".format(rd))
            # initialize arms
            self.arms = self.set_up_arms()
            for agent in self.agents:
                agent.get_current_arms_group(self.get_current_arms_group())
                agent.get_current_feature_vector(self.get_current_feature_vector())
                # Select arm and pull it
                chosen_arm = agent.select_arm()
                reward = self.arms[chosen_arm].pull()
                agent.update(rd, chosen_arm, reward)
                print("agent: ", agent.id, "chosen arm: ", chosen_arm)
        
        print("========== Game End ===========")
        # Calculate the average reward
        for agent in self.agents:
            print("Agent reward:",  agent.get_all_reward())


### code for test ###
n_arms = 3
round = 10
m_agents = 2
g_groups = 2
feature_vector_length = 5
env = Environment(n_arms, round, m_agents, g_groups, feature_vector_length)      
env.run_simulation()
