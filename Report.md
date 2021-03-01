# Project Report

## Learning Algorithm


## Implementation

#### Model architecture for the neural network:
Actor
- input: 33, output: 128 (ReLu activation)
- input: 128, output: 128 (ReLu activation)
- input: 128, output: 4 (tanh activation)
        
Critic
- input: 33, output: 128 (ReLu activation)
- input: 128+4, output: 128 (ReLu activation)
- input: 128, output: 1 (No activation) 

#### Steps taken to generate the final network:

#### Hyperparameters used in the final training solution:


## Results

Episode 100	Average Score: 0.01
Episode 200	Average Score: 0.02
Episode 300	Average Score: 0.02
Episode 400	Average Score: 0.00
Episode 500	Average Score: 0.02
Episode 600	Average Score: 0.06
Episode 700	Average Score: 0.11
Episode 800	Average Score: 0.37
Episode 900	Average Score: 0.22
Episode 952	Score: 2.10	Running Average: 0.50
Environment solved in 952 episodes!	Average Score: 0.50

## Future Work
