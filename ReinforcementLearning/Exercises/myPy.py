import math
import keras
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm.notebook import tqdm
from IPython.display import display, HTML

pd.set_option("display.max_rows", None)
def display_df(df):
    display(HTML("<div style='height: 200px; overflow: auto; width: fit-content'>" + df.to_html() + "</div>"))

keras.utils.set_random_seed(42)

# Download Sample Data
data = pd.read_csv('AAPL_2009_4m_training_features_1d.csv')

## 2. Train / Test Split
training_rows = int(len(data) * 0.8)
train_df = data.iloc[:training_rows].set_index("Date")
test_df = data.iloc[training_rows:].set_index("Date")

# By defining the indices on train_df, they will match the structure of X_train.
idx_close = train_df.columns.get_loc('Close')        # Correctly returns 0
idx_bb_upper = train_df.columns.get_loc('BB_upper')  # Correctly returns 1
idx_bb_lower = train_df.columns.get_loc('BB_lower')  # Correctly returns 2

# convert train and test dfs to np arrays with dtype=float
X_train = train_df.values.astype(float)
X_test = test_df.values.astype(float)

## 3. Define the Agent
@keras.saving.register_keras_serializable()
class DQN(keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        model = keras.Sequential([
            keras.Input(shape=(state_size,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(action_size, activation="linear")
        ])        
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001))
        self.model = model

class Agent:
    def __init__(self, window_size, num_features, test_mode=False, model_name=''):
        self.window_size = window_size
        self.num_features = num_features
        self.state_size = window_size * num_features
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.test_mode = test_mode
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = keras.models.load_model(model_name) if test_mode else self._model()

    def _model(self):
        model = DQN(self.state_size, self.action_size).model
        return model

    def get_q_values_for_state(self, state):
        return self.model.predict(state.flatten().reshape(1, self.state_size), verbose=0)

    def fit_model(self, input_state, target_output):
        return self.model.fit(input_state.flatten().reshape(1, self.state_size), target_output, epochs=1, verbose=0)    

    def act(self, state): 
        if not self.test_mode and random.random() <= self.epsilon:
            return random.randrange(self.action_size)   
        q_values = self.get_q_values_for_state(state)
        return np.argmax(q_values[0]) 

    def exp_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        mini_batch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                future_q_values = self.get_q_values_for_state(next_state)
                target = reward + self.gamma * np.amax(future_q_values[0])
            target_q_table = self.get_q_values_for_state(state)  
            target_q_table[0][action] = target
            history = self.fit_model(state, target_q_table)
            losses.extend(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return losses

# Initialize agent
window_size = 1
agent = Agent(window_size, num_features=X_train.shape[1])

def format_price(n):
    return ('-$' if n < 0 else '$') + '{0:.2f}'.format(abs(n))

sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))

def plot_behavior(data_input, bb_upper_data, bb_lower_data, states_buy, states_sell, profit, train=True):
    fig = plt.figure(figsize = (15,5))
    plt.plot(data_input, color='k', lw=2., label= 'Close Price')
    plt.plot(bb_upper_data, color='b', lw=2., label = 'Bollinger Bands')
    plt.plot(bb_lower_data, color='b', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='g', label = 'Buying signal', markevery = states_buy)
    plt.plot(data_input, 'v', markersize=10, color='r', label = 'Selling signal', markevery = states_sell)
    plt.title(f'Total gains: {format_price(profit)}')
    plt.legend()
    plt.show()

def plot_losses(losses, title):
    plt.plot(losses)
    plt.title(title)
    plt.ylabel('MSE Loss Value')
    plt.xlabel('Batch Number')
    plt.show()

def get_state(data, t, n):
    start = t - n + 1
    end = t + 1
    if start < 0:
        padding = np.tile(data[0], (abs(start), 1))
        actual_data = data[0:end]
        block = np.vstack((padding, actual_data))
    else:
        block = data[start:end]
    res = sigmoid(block)
    return res

keras.config.disable_traceback_filtering()
l = len(X_train) - 1
batch_size = 32
episode_count = 2

batch_losses = []
total_episodes_trained = 0

for e in range(episode_count):
    print(f"\n--- Starting Episode: {e+1}/{episode_count} ---")
    state = get_state(X_train, 0, window_size)
    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []

    for t in tqdm(range(l), desc=f'Running episode {e+1}/{episode_count}'):
        action = agent.act(state)
        next_state = get_state(X_train, t + 1, window_size)
        reward = 0

        if action == 1: # Buy
            agent.inventory.append(X_train[t, idx_close])
            states_buy.append(t)
        elif action == 2 and len(agent.inventory) > 0: # Sell
            bought_price = agent.inventory.pop(0)
            sell_price = X_train[t, idx_close]
            trade_profit = sell_price - bought_price
            reward = max(trade_profit, 0)
            total_profit += trade_profit
            states_sell.append(t)

        done = t == l - 1
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) > batch_size:
            losses_for_batch = agent.exp_replay(batch_size)
            batch_losses.extend(losses_for_batch)

    print(f'--------------------------------\nEpisode {e+1} Summary\nTotal Profit: {format_price(total_profit)}\n--------------------------------')
    plot_behavior(X_train[:, idx_close], X_train[:, idx_bb_upper], X_train[:, idx_bb_lower], states_buy, states_sell, total_profit)
    plot_losses(batch_losses[total_episodes_trained:], f'Episode {e+1} DQN Model Loss')
    total_episodes_trained = len(batch_losses)
    agent.model.save(f'model_ep{e+1}.keras')

plot_losses(batch_losses, "Total Training Loss Across All Episodes")

l_test = len(X_test) - 1
total_profit = 0
states_sell_test = []
states_buy_test = []

agent = Agent(window_size, num_features=X_test.shape[1], test_mode=True, model_name=f'model_ep{episode_count}.keras')
agent.inventory = []
state = get_state(X_test, 0, window_size)

for t in tqdm(range(l_test), desc="Running Test"):
    action = agent.act(state)
    next_state = get_state(X_test, t + 1, window_size)

    if action == 1: # Buy
        buy_price = X_test[t, idx_close]
        agent.inventory.append(buy_price)
        states_buy_test.append(t)
    elif action == 2 and len(agent.inventory) > 0: # Sell
        bought_price = agent.inventory.pop(0)
        sell_price = X_test[t, idx_close]
        trade_profit = sell_price - bought_price
        total_profit += trade_profit
        states_sell_test.append(t)

    state = next_state

print('------------------------------------------')
print(f'Total Test Profit: {format_price(total_profit)}')
print('------------------------------------------')

plot_behavior(X_test[:, idx_close], X_test[:, idx_bb_upper], X_test[:, idx_bb_lower], states_buy_test, states_sell_test, total_profit, train=False)