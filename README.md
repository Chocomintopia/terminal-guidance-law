# DESIGN OF TERMINAL GUIDANCE LAW BASED ON SPARSE REWARD REINFORCE LEARNING

## Prioritized Experience Replay (PER)

`airdensity.py` : about aerodynamics, from DOI:10.27061/d.cnki.ghgdu.2021.003352

`table.py` : about aerodynamics, from DOI:10.27061/d.cnki.ghgdu.2021.003352

`testEnv.py` : environment, from DOI:10.27061/d.cnki.ghgdu.2021.003352

`dqn_train.py` : run this to train DQN terminal guidance law, from DOI:10.27061/d.cnki.ghgdu.2021.003352

`dqn_test.py` : run this to test DQN terminal guidance law, from DOI:10.27061/d.cnki.ghgdu.2021.003352

`per_sumtree.py` : sumtree, from [Howuhh/prioritized_experience_replay](https://github.com/Howuhh/prioritized_experience_replay)

`per_buffer.py` : per buffer, from [Howuhh/prioritized_experience_replay](https://github.com/Howuhh/prioritized_experience_replay)

`per_train.py` : run this to train PER terminal guidance law, references [Howuhh/prioritized_experience_replay](https://github.com/Howuhh/prioritized_experience_replay)

`per_test.py` : run this to test PER terminal guidance law,

#### Results

![result](/result.svg)

#### References

environment from DOI : 10.27061/d.cnki.ghgdu.2021.003352

PER from : [Howuhh/prioritized_experience_replay](https://github.com/Howuhh/prioritized_experience_replay)



## Hindsight Experience Replay (HER)

`airdensity.py` : about aerodynamics, from DOI:10.27061/d.cnki.ghgdu.2021.003352

`table.py` : about aerodynamics, from DOI:10.27061/d.cnki.ghgdu.2021.003352

`testEnv.py` : environment, from DOI:10.27061/d.cnki.ghgdu.2021.003352

`hindsightEnv.py` : to “hindsight” calculate state using the trajectory of the missile and the target

`hindsight_dqn_train.py` : run this to train Hindsight terminal guidance law

`hindsight_dqn_test.py` : run this to test Hindsight terminal guidance law