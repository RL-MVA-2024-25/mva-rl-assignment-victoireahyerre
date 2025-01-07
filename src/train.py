from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from collections import deque
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 



class ProjectAgent:
    def __init__(self, env = env, model=None, gamma=0.99, buffer_size=10000, min_samples=100):
        self.env = env
        self.gamma = gamma # Discount factor
        self.replay_buffer = deque(maxlen=buffer_size) # Replay buffer
        self.model = model if model else RandomForestRegressor(n_estimators=200) # Model
        self.min_samples = min_samples # Minimum number of samples to train the model
        self.fitted = False # Set to True after the first training
        self.save()

    def act(self, observation, use_random=False):
        """Select an action based on the observation."""
        state = np.array(observation).reshape(1, -1)
        if self.fitted and not use_random:
            q_values = np.array([self.model.predict(np.hstack([state, [[a]]])) for a in range(self.env.action_space.n)]).flatten()
            return np.argmax(q_values)
        return self.env.action_space.sample()  # Random action

    def store_transition(self, state, action, reward, next_state, done, trunc):
        """Store a transition in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done, trunc))

    def train(self):
        """Train the model on the replay buffer."""
        if len(self.replay_buffer) < self.min_samples:
            return # Not enough samples to train the model

        X, y = [], [] # Features and targets
        for state, action, reward, next_state, done, trunc in self.replay_buffer: # Iterate over the replay buffer
            state_action = np.hstack([state, action]).reshape(1, -1) # State-action pair

            if done or trunc: # Terminal state
                target = reward
            else:
                if self.fitted: # Compute the target using the model
                    next_q_values = np.array([self.model.predict(np.hstack([next_state, [a]]).reshape(1, -1)) for a in range(self.env.action_space.n)]).flatten()
                    target = reward + self.gamma * np.max(next_q_values)
                else:
                    target = reward  

            X.append(state_action.flatten())
            y.append(target)

        self.model.fit(np.array(X), np.array(y)) # Train the model
        self.fitted = True

    def save(self, path="fqi_agent.pkl"):
        """Save the model after training it."""
        for episode in range(300):
            state, _ = env.reset()
            done, trunc = False, False
            while not (done or trunc):
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                self.store_transition(state, action, reward, next_state, done, trunc)
                state = next_state
            self.train()
            print(f"Episode {episode + 1}")
        print("Model trained")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print("Model saved")

    def load(self, path="fqi_agent.pkl"):
        """Load the model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print("Model loaded")
