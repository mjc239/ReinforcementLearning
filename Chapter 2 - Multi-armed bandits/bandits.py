import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import stats


class GaussianArm:
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def pull(self):
        return self.mean + self.std*np.random.randn()


class OrnsteinUhlenbeckArm:
    
    def __init__(self, std, center, reversion_speed, volatility):
        self.center = center
        self.reversion_speed = reversion_speed
        self.volatility = volatility
        
        self.mean = center
        self.std = std
        
    def ou_update(self):
        self.mean += self.reversion_speed*(self.center - self.mean) + self.volatility*np.random.randn()
        
    def pull(self):
        return self.mean + self.std*np.random.randn()

    
class ArmBed:
    
    def __init__(self, means, stds, reversion_speeds=None, volatilities=None):
        self.means = means
        self.stds = stds
        self.reversion_speeds = reversion_speeds
        self.volatilities = volatilities
        assert len(means) == len(stds)
        assert (reversion_speeds is None) == (volatilities is None)
        if reversion_speeds is not None:
            assert (len(reversion_speeds), len(volatilities)) == (len(means), len(means)), f'{len(reversion_speeds)}, {len(volatilities)}, {len(means)}'
            self.arms = [OrnsteinUhlenbeckArm(std, center, rev, vol) for std, center, rev, vol in zip(stds, means, reversion_speeds, volatilities)]
        else:
            self.arms = [GaussianArm(mean, std) for mean, std in zip(means, stds)]
        
    def plot_bed(self):
        x = np.linspace(min(self.means)-3*self.stds[np.argmin(self.means)], max(self.means) + 3*self.stds[np.argmax(self.means)], 1000)
        for i, arm in enumerate(self.arms):
            plt.plot(x, stats.norm.pdf(x, arm.mean, arm.std), label=f'Arm {i}')
        plt.title('Distribution of returns from arms in bed')
        plt.xlabel('Reward')
        plt.ylabel('Probability density')
        plt.legend()
        
        
class Bandit(ArmBed):
    
    def __init__(self, arm_mean, arm_std, num_arms, initial_values=0, reversion_speeds=None, volatilities=None):
        self.num_arms = num_arms
        self.initial_values = initial_values
        
        # Arm means sampled from N(arm_mean, arm_std^2) and arm stds = 1
        means = arm_mean + arm_std*np.random.randn(self.num_arms)
        super().__init__(means, np.ones(self.num_arms), reversion_speeds, volatilities)
        
        self.Q = self.initial_values*np.ones(self.num_arms)
        self.N = np.zeros(self.num_arms)
        self.R = 0
        self.t = 1
        
    def basic_update(self, reward, arm, alpha):
        self.R += reward
        self.N[arm] += 1
        if alpha:
            self.Q[arm] += (reward - self.Q[arm])*alpha
        else:
            self.Q[arm] += (reward - self.Q[arm])/self.N[arm]
        
        if self.reversion_speeds is not None:
            for arm in self.arms:
                arm.ou_update()
           
    
class EpsGreedyBandit(Bandit):
    
    def __init__(self, arm_mean, arm_std, num_arms, epsilon, initial_values=0, alpha=None, reversion_speeds=None, volatilities=None):
        super().__init__(arm_mean, arm_std, num_arms, initial_values, reversion_speeds, volatilities)
        self.epsilon = epsilon
        self.alpha = alpha
        
    def select_arm(self, update=True):
        # Choose arm to pull
        if np.random.rand() < self.epsilon:
            arm = np.random.choice(range(self.num_arms))
        else:
            arm = np.argmax(self.Q)
    
        # Pull arm for reward
        reward = self.arms[arm].pull()
        
        # Update Q/R/N
        if update:
            self.basic_update(reward, arm, self.alpha)

        return reward
    
    def simulate(self, num_steps):
        rewards = np.zeros(num_steps)
        for step in range(num_steps):
            rewards[step] = self.select_arm()
        return rewards


class UCBBandit(Bandit):
    
    def __init__(self, arm_mean, arm_std, num_arms, c, initial_values=0, alpha=None, reversion_speeds=None, volatilities=None):
        super().__init__(arm_mean, arm_std, num_arms, initial_values, reversion_speeds, volatilities)
        self.c = c
        self.alpha = alpha
    
    def select_arm(self, update=True):
        # Choose arm to pull
        arm = np.argmax(self.Q + self.c*np.sqrt(np.log(self.t)/(self.N + 1e-20)))
        
        # Pull arm for reward
        reward = self.arms[arm].pull()
                
        # Update Q/R/N
        if update:
            self.basic_update(reward, arm, self.alpha)
            self.t += 1
            
        return reward
    
    def simulate(self, num_steps):
        rewards = np.zeros(num_steps)
        for step in range(num_steps):
            rewards[step] = self.select_arm()
        return rewards


class GradientBandit(Bandit):
    
    def __init__(self, arm_mean, arm_std, num_arms, alpha, reversion_speeds=None, volatilities=None):
        super().__init__(arm_mean, arm_std, num_arms, 0, reversion_speeds, volatilities)
        self.alpha = alpha
        self.baseline = 0
        
    def compute_probs(self):
        exp_Q = np.exp(self.Q)
        return exp_Q/np.sum(exp_Q)
    
    def select_arm(self, update=True):
        # Choose arm to pull
        probs = self.compute_probs()
        arm = np.random.choice(np.arange(self.num_arms), p=probs)
        
        # Pull arm for reward
        reward = self.arms[arm].pull()
        
        # Update Q/R/N
        if update:
            self.R += reward
            avg_reward = self.R/self.t

            self.N[arm] += 1
            self.Q += self.alpha*(reward - avg_reward)*(np.array([(i == arm) for i in range(self.num_arms)]) - probs)
            self.t += 1
            
            if self.reversion_speeds is not None:
                for arm in self.arms:
                    arm.ou_update()
            
        return reward
    
    def simulate(self, num_steps):
        rewards = np.zeros(num_steps)
        for step in range(num_steps):
            rewards[step] = self.select_arm()
        return rewards


def epsilon_greedy_performance(arm_mean, arm_std, epsilon, alpha=None, num_runs=3000, steps_per_run=1000, initialisation=0, reversion_speeds=None, volatilities=None):
    average_rewards = np.zeros(steps_per_run)
    
    for i in tqdm.tqdm(range(num_runs)):
        bandit = EpsGreedyBandit(arm_mean,
                                 arm_std,
                                 10,
                                 epsilon,
                                 alpha=alpha,
                                 initial_values=initialisation,
                                 reversion_speeds=reversion_speeds,
                                 volatilities=volatilities)
        rewards = bandit.simulate(steps_per_run)
        average_rewards += np.array(rewards)
    average_rewards /= num_runs
    
    return average_rewards


def ucb_performance(arm_mean, arm_std, c, num_runs=3000, steps_per_run=1000, initialisation=0, reversion_speeds=None, volatilities=None):
    average_rewards = np.zeros(steps_per_run)
    
    for i in tqdm.tqdm(range(num_runs)):
        bandit = UCBBandit(arm_mean, 
                           arm_std, 
                           10, 
                           c, 
                           initial_values=initialisation, 
                           reversion_speeds=reversion_speeds, 
                           volatilities=volatilities)
        rewards = bandit.simulate(steps_per_run)
        average_rewards += np.array(rewards)
    average_rewards /= num_runs
    
    return average_rewards


def grad_bandit_performance(arm_mean, arm_std, alpha, num_runs=3000, steps_per_run=1000, reversion_speeds=None, volatilities=None):
    average_rewards = np.zeros(steps_per_run)
    
    for i in tqdm.tqdm(range(num_runs)):
        bandit = GradientBandit(arm_mean, 
                                arm_std, 
                                10, 
                                alpha, 
                                reversion_speeds=reversion_speeds, 
                                volatilities=volatilities)
        rewards = bandit.simulate(steps_per_run)
        average_rewards += np.array(rewards)
    average_rewards /= num_runs
    
    return average_rewards