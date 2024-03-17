import numpy as np

'''
code for the bandit simulation
    - single arm
'''

# Set parameters for the normal distribution
mean = 0    # Mean value
stddev = 1  # Standard deviation
#size = 10   # Number of random values to generate, equal to total round

class ArmGroup:
    def __init__(self,group_id, group_coefficients):
        self.id = group_id
        self.arms = []
        self.group_coefficients = group_coefficients

    def create_new_arm(self, arm_id, feature_vector):
        new_arm = Arm(arm_id, self.group_id, feature_vector, self.group_coefficients, np.random.normal(mean, stddev))
        # self.arms.append(new_arm)
        return new_arm
    
    def get_coefficients(self):
        return self.group_coefficients

class Arm:
    def __init__(self, arm_id, group_id, feature_vector, group_coefficients, distribution):
        self.arm_id = arm_id
        self.group_id = group_id
        self.feature_vector = feature_vector
        self.group_coefficients = group_coefficients
        # Generate random variables that follow a normal distribution
        self.distribution = distribution

    def pull(self):
        return np.dot(self.feature_vector, self.group_coefficients) + self.distribution
    
    def get_feature_vector(self):
        return self.feature_vector
    
    def get_group(self):
        return self.group_id