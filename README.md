# ML-Agents Collaboration and Competition
![gif](images/trained.gif)

## Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

## Dependencies
1. Clone this repository to your local drive.

2. Download the environment from one of the links below and extract in the repository directory:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

3. Navigate to the repository directory then create and activate a new environment with Python 3.6:
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```  

4. Install the required packages torch, numpy and unityagents. They can be installed using pip:
    ```
    pip install torch numpy unityagents
    ```

## Running the Project
With the environment activated run `train.py` to begin training. 

The trained weights are saved to the files `checkpoint_actor.pth` and `checkpoint_critic.pth` once the required score is reached.
