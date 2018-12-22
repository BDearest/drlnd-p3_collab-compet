## Project Details
This project trains two agents to play tennis. The state has a length of 24 for the agents with 8 variables for the past three timesteps that describe the position and speed of both the ball and the racket. The action space is continuous, basically all the positions between the net and the end of the court for each agent plus jumping. Rewards are a score of +0.1 for getting the ball over the net and -0.01 if the ball hits the ground or goes out of bounds. The environment is considered solved when an average score of 0.5 is achieved over 100 episodes. The score for each episode is determined by taking the maximum of the two agent's scores.

Note that the agent and model code in this repository is a modified version of the [DDPG-Bipedal](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) implementation in the Deep Reinforcement Learning Nanodegree repository on Github.

## Getting Started

In order to run this agent, one must have Python 3, PyTorch, Unity, Jupyter Notebook, numpy, and matplotlib installed. The easiest way to set up all of the requirements and dependencies is to clone the Deep Reinforcement Learning Nanodegree repository on Github. One will need to create a new virtual environment with Python 3.6, install OpenAI gym and create a kernel with IPython. Once the notebook is open, the kernel should be selected from the kernel dropdown menu. Detailed instructions for these steps can be found on the [Deep Reinforcement Learning Nanodegree Github](https://github.com/udacity/deep-reinforcement-learning#dependencies). Finally, one would need to set up the Tennis environment by downloading the appropriate environment for one's operating system from the links below.
* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* [Linux (headless)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
* [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Once the environment is downloaded, it should be put in the root directory of this cloned repository and unzipped.

To run the code, open Tennis.ipynb in Jupyter Notebook and run all cells.
