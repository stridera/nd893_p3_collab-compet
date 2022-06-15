[//]: # "Image References"
[scores]: images/graph.png "Training Score Graph"

# Training Report

## Deep Deterministic Policy Gradient (DDPG)

DDPG is a off-policy model designed for continuous actions. (Actions that aren't simply up, down, left right, but is
more like 0.2 to the left, and 0.8 up.)

## Multi-Agent DDPG (MADDPG)

Multi-Agent DDPG runs similar to DDPG does but the critics looks at the whole environment instead of just the results of the single action. It attempts to
critique the gameplay in a way that is beneficial to all agents instead of just itself.

### Model Architecture

The network followed the paper with 3 linear layers linked with ReLU (Rectified Linear Unit) activation functions.

The model input and output were obtained directly from the environment.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry is clipped between
-1 and 1.

So the final trained models network looks like this:

```python
Actor(
  (fc1): Linear(in_features=33, out_features=400, bias=True)
  (bn1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=400, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=4, bias=True)
)
Critic(
  (fcs1): Linear(in_features=33, out_features=400, bias=True)
  (bn1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=404, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=1, bias=True)
)
```

The Actor returns 2 actions while the critics returns a boolean determining if it thinks the action is a good move or not.

We use an Adam optimizer to train the models and a mean square function for determining loss.

Other noticable actions we do include adding noise to the action to give some randomness in the training.

### Hyper-parameters

The environment was solved using the DDPG model with the following hyper-parameters:

- Buffer Size: `1,000,000` We keep up to 1M samples around to train the model with.
- Batch Size: `512` Choosing a batch size is difficult. I selected a higher batch size for this.
- Gamma: `0.99` The gamma factor is used to determine the reward discount. This will slowly discount the reward to make it so the chain of actions leading toward a positive reward are recognized.
- Tau: `0.001` The tau value is used to soft update the target network. This means we only slowly update the target network using the following update schedule: *θ*−=*θ*×*τ*+*θ*−×(1−*τ*)
- LR_ACTOR: `0.0004` - The Actor learning rate.
- LR_CRITIC: `0.004` - The Critic Learning Rate. Played around with a bunch of values and this worked well. Should be kept
  higher than or equal to the actor learning rate.
- LEARN_EVERY: `1` - Learns on every timestep.

I was never able to get the MADDPG model to converge. I've tried training on buffer sizes between 64-1028, different noise profiles, and learning rates.
I'm unsure if it's due to a bug or just the complex nature of trying to fuse both observations into one critic.

## Results

The model trained quickly, reaching a solved status (+0.5/100eps) at around episode 1500 (Around 1.5 hours training.) There was a lot of fluxuation in this
environment that made the scores jump around.

![Scores Graph][scores]

## Future Improvements

DDPG worked well enough for this environment. The model was trained to handle both sides and it actually worked pretty well.
I can't help but think figuring out the MADDPG approach would train faster, but 1.5 hours to reach the solved state was pretty
good as it was.

I think using a prioritized replay buffer would help some as well, as well as more reward shaping.
