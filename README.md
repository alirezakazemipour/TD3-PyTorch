[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)  
# TD3-PyTorch
TD3 (Twin Delayed Deep Deterministic Policy Gradient) is somehow the equivalent of Double Q-Learning in Continuous Domain. To avoid overestimation existed in DDPG method, a similar trick from Double Q-Learning is used: Disentangle **Action Selection** part from **Action Evaluation** in the Bellman Equation by using two separate networks.  
TD3 also uses some tricks (like **Target Policy Smoothing** ) to reduce high variance common in Policy Gradient methods.


## Results
> x-axis: episode number.

Ant| Hopper
:-----------------------:|:-----------------------:|
![](results/ant.png)| ![](results/hopper.png)

## Dependencies
- gym == 0.17.3
- mujoco-py == 2.0.2.13
- numpy == 1.19.2
- opencv_contrib_python == 4.4.0.44
- psutil == 5.5.1
- torch == 1.6.0

## Installation
```shell
pip3 install -r requirements.txt
```


## References
- [_Addressing Function Approximation Error in Actor-Critic Methods_, Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)

## Acknowledgement 
- Big thanks to [@sfujim](https://github.com/sfujim) for [TD3](https://github.com/sfujim/TD3).