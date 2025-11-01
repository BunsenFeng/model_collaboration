import os
import math
import json
import torch
import shutil
import random

class NumericSwarm:

    def __init__(self, dimension, population, starting_velocity_mode="random", weight_randomness=True, inertia=0.2, cognitive_coeff=0.3, social_coeff=0.4, repel_coeff=0.05, step_length=0.5, repel_term=True, step_length_factor=0.95, minimum_step_length=0.1, patience=5, restart_patience=3):
        
        self.dimension = dimension
        self.population = population
        self.starting_velocity_mode = starting_velocity_mode
        self.weight_randomness = weight_randomness
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.repel_coeff = repel_coeff
        self.step_length = step_length
        self.repel_term = repel_term
        self.step_length_factor = step_length_factor
        self.minimum_step_length = minimum_step_length
        self.patience = patience
        self.restart_patience = restart_patience

        self.particles = []
        for _i in range(population):
            position = [random.uniform(-1, 1) for _ in range(dimension)]
            self.particles.append(torch.tensor(position, dtype=torch.float32))
        
        # initialize utility scratchpad
        self.utility_scratchpad = {"g": None, "g_worst": None, "g_history": []}
        for i in range(len(self.particles)):
            self.utility_scratchpad[f"model_{i}_now"] = None
            self.utility_scratchpad[f"model_{i}_best"] = None
            self.utility_scratchpad[f"model_{i}_history"] = []

        # initialize personal bests
        self.personal_bests = self.particles.copy()
        self.personal_best_scores = [float('-inf')] * population

        # initialize velocities
        self.velocities = []
        for _i in range(population):
            if self.starting_velocity_mode == 'zero':
                velocity = [0.0 for _ in range(dimension)]
            elif self.starting_velocity_mode == 'random':
                velocity = [random.uniform(-1, 1) for _ in range(dimension)]
            else:
                raise ValueError("Invalid starting_velocity_mode")
            self.velocities.append(torch.tensor(velocity, dtype=torch.float32))
        
        # intialize global best and global worst
        self.global_best = random.choice(self.particles)
        self.global_best_score = float('-inf')
        self.global_worst = random.choice(self.particles)
        self.global_worst_score = float('inf')

    def update(self, scores):
        assert len(scores) == self.population, "Scores length must match population size"

        # update utility scratchpad
        for i in range(len(self.particles)):
            self.utility_scratchpad[f"model_{i}_now"] = scores[i]
            self.utility_scratchpad[f"model_{i}_history"].append(scores[i])
            if self.utility_scratchpad[f"model_{i}_best"] is None or scores[i] > self.utility_scratchpad[f"model_{i}_best"]:
                self.utility_scratchpad[f"model_{i}_best"] = scores[i]
                self.personal_bests[i] = self.particles[i]
        
        if self.utility_scratchpad["g"] is None or max(scores) > self.utility_scratchpad["g"]:
            self.utility_scratchpad["g"] = max(scores)
            self.global_best = self.particles[scores.index(max(scores))]
            self.global_best_score = max(scores)

        if self.utility_scratchpad["g_worst"] is None or min(scores) < self.utility_scratchpad["g_worst"]:
            self.utility_scratchpad["g_worst"] = min(scores)
            self.global_worst = self.particles[scores.index(min(scores))]
            self.global_worst_score = min(scores)
        
        self.utility_scratchpad["g_history"].append(self.utility_scratchpad["g"])

        # if "g_history" did not improve for 'patience' iterations, terminate signal
        if len(self.utility_scratchpad["g_history"]) > self.patience:
            recent_history = self.utility_scratchpad["g_history"][-self.patience:]
            if self.utility_scratchpad["g_history"][-1] <= max(recent_history[:-1]):
                return True  # terminate signal

        for i in range(self.population):
            # judge restart flag
            if len(self.utility_scratchpad[f"model_{i}_history"]) > self.restart_patience:
                recent_history = self.utility_scratchpad[f"model_{i}_history"][-self.restart_patience:]
                if self.utility_scratchpad[f"model_{i}_history"][-1] <= max(recent_history[:-1]):
                    # restart particle
                    self.particles[i] = self.personal_bests[i]
                    self.velocities[i] = torch.tensor([random.uniform(-1, 1) for _ in range(self.dimension)], dtype=torch.float32)
                    continue  # skip velocity and position update this iteration

            # weight_randomness
            if self.weight_randomness:
                r_w = random.uniform(0, 1)
                r_p = random.uniform(0, 1)
                r_s = random.uniform(0, 1)
                r_b = random.uniform(0, 1) # b for bad, repel term weight
            else:
                r_w = 1
                r_p = 1
                r_s = 1
                r_b = 1

            # weight normalize
            self_weight = self.inertia * r_w
            cognitive_weight = self.cognitive_coeff * r_p
            social_weight = self.social_coeff * r_s
            repel_weight = self.repel_coeff * r_b if self.repel_term else 0.0
            weight_sum = self_weight + cognitive_weight + social_weight + repel_weight

            self_weight /= weight_sum
            cognitive_weight /= weight_sum
            social_weight /= weight_sum
            repel_weight /= weight_sum

            # update velocity
            self.velocities[i] = (
                self.velocities[i] * self_weight +
                (self.personal_bests[i] - self.particles[i]) * cognitive_weight +
                (self.global_best - self.particles[i]) * social_weight +
                (self.particles[i] - self.global_worst) * repel_weight
            )

            # update position
            self.particles[i] = self.particles[i] + self.velocities[i] * self.step_length

        # update step length
        self.step_length *= self.step_length_factor
        self.step_length = max(self.step_length, self.minimum_step_length)

        return False # no terminate signal

    def get_particles(self):
        return self.particles

    def get_global_best_particle(self):
        return self.global_best

if __name__ == '__main__':
    swarm = NumericSwarm(
        dimension=3,
        population=5,
        starting_velocity_mode='random',
        weight_randomness=True,
        inertia=0.5,
        cognitive_coeff=1.5,
        social_coeff=1.5,
        repel_coeff=1.0,
        step_length=0.1,
        repel_term=True,
        step_length_factor=0.95,
        minimum_step_length=0.01,
        patience=10,
        restart_patience=5
    )

    for iter in range(50):
        scores = []
        for particle in swarm.get_particles():
            # Example objective function: -(x-0.5)^2 -(y-0.5)^2 -(z-0.5)^2
            score = -sum((x - 0.5) ** 2 for x in particle.tolist())
            scores.append(score)
        terminate = swarm.update(scores)
        print(f"Iteration {iter+1}, Global Best Score: {swarm.global_best_score}")
        if terminate:
            print("Termination condition met.")
            break

    # swarm.update([0.1, 0.5, 0.3, 0.7, 0.2])
    # print(swarm.get_global_best_particle())
    # swarm.update([0.2, 0.6, 0.4, 0.8, 0.3])
    # swarm.update([0.3, 0.7, 0.5, 0.4, 0.9])
    # print(swarm.get_global_best_particle())
    # print(swarm.get_particles())