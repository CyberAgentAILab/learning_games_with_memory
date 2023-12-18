# Learning in Multi-Memory Games Triggers Complex Dynamics Diverging from Nash Equilibrium
Code for reproducing results in the paper "[Learning in Multi-Memory Games Triggers Complex Dynamics Diverging from Nash Equilibrium](https://arxiv.org/abs/2302.01073)". [Published in IJCAI](https://www.ijcai.org/proceedings/2023/14).

## About
Repeated games consider a situation where multiple agents are motivated by their independent rewards throughout learning. In general, the dynamics of their learning become complex. Especially when their rewards compete with each other like zero-sum games, the dynamics often do not converge to their optimum, i.e., the Nash equilibrium. To tackle such complexity, many studies have understood various learning algorithms as dynamical systems and discovered qualitative insights among the algorithms. However, such studies have yet to handle multi-memory games (where agents can memorize actions they played in the past and choose their actions based on their memories), even though memorization plays a pivotal role in artificial intelligence and interpersonal relationship. This study extends two major learning algorithms in games, i.e., replicator dynamics and gradient ascent, into multi-memory games. Then, we prove their dynamics are identical. Furthermore, theoretically and experimentally, we clarify that the learning dynamics diverge from the Nash equilibrium in multi-memory zero-sum games and reach heteroclinic cycles (sojourn longer around the boundary of the strategy space), providing a fundamental advance in learning in games.

## Installation
This code is written in Python 3. To install the required dependencies, execute the following command:
```bash
$ pip install numpy
```

## How to use
"cont_MMGA.py" outputs the data necessary to draw Fig. 2 and 3.
All of "disc_MMGA_Xac_Ymm", where X (resp. Y) indicates the number of actions (resp. memories), output the data necessary to draw Fig. 4.

## Citation
If you use our code in your work, please cite our paper:
```
@inproceedings{fujimoto2023learning,
  title     = {Learning in Multi-Memory Games Triggers Complex Dynamics Diverging from Nash Equilibrium},
  author    = {Fujimoto, Yuma and Ariu, Kaito and Abe, Kenshi},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {118--125},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/14},
  url       = {https://doi.org/10.24963/ijcai.2023/14},
}
```
