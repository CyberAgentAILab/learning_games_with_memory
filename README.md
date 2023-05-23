# Learning in Multi-Memory Games Triggers Complex Dynamics Diverging from Nash Equilibrium
Code for reproducing results in the paper "[Learning in Multi-Memory Games Triggers Complex Dynamics Diverging from Nash Equilibrium](https://arxiv.org/abs/2302.01073)".

## About
Repeated games consider a situation where multiple agents are motivated by their independent rewards throughout learning. In general, the dynamics of their learning become complex. Especially when their rewards compete with each other like zero-sum games, the dynamics often do not converge to their optimum, i.e., the Nash equilibrium. To tackle such complexity, many studies have understood various learning algorithms as dynamical systems and discovered qualitative insights among the algorithms. However, such studies have yet to handle multi-memory games (where agents can memorize actions they played in the past and choose their actions based on their memories), even though memorization plays a pivotal role in artificial intelligence and interpersonal relationship. This study extends two major learning algorithms in games, i.e., replicator dynamics and gradient ascent, into multi-memory games. Then, we prove their dynamics are identical. Furthermore, theoretically and experimentally, we clarify that the learning dynamics diverge from the Nash equilibrium in multi-memory zero-sum games and reach heteroclinic cycles (sojourn longer around the boundary of the strategy space), providing a fundamental advance in learning in games.

## How to use
"cont_MMGA.py" outputs the data necessary to draw Fig. 2 and 3.

All of "disc_MMGA_Xac_Ymm", where X (resp. Y) indicates the number of actions (resp. memories), output the data necessary to draw Fig. 4.

## Citation
If you use our code in your work, please cite our paper:
```
@article{fujimoto2023learning,
  title={Learning in Multi-Memory Games Triggers Complex Dynamics Diverging from Nash Equilibrium},
  author={Fujimoto, Yuma and Ariu, Kaito and Abe, Kenshi},
  journal={arXiv preprint arXiv:2302.01073},
  year={2023}
}
```
