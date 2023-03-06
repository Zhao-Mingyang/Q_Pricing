# Q_Pricing

## Run the code

### Standard Q learning Training:
```
python train.py 
```
The program runs the Q learning for 1000000 iterations `total = 100` times. Please change the `self.EPSILON` in Q_agent.py to replicate the result with different epsilon and uncomment the line 75 `+ self.ALPHA * np.random.uniform(-25/1000, 25/1000)` in Q_agent.py to add noise to the Q values.

### Q learning Training with Predicting Opponent's Actions:
```
python Policy_train.py
```
Please change the variable `continueous_price` to `False` for discrete price stragedy with price intervals of 100.
