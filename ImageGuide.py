from torch import optim, nn, randn_like, zeros_like, rand_like
from pytti.Notebook import tqdm
from pytti import *
import pandas as pd
import math
import matplotlib.pyplot as plt

from labellines import labelLines
from scipy.signal import savgol_filter

def unpack_dict(D, n=2):
    ds = [{k: V[i] for k, V in D.items()} for i in range(n)]
    return tuple(ds)

def smooth_dataframe(df, window_size):
    """Applies a moving average filter to the columns of df."""
    smoothed_df = pd.DataFrame(index=df.index, columns=df.columns)
    for key in df.columns:
        smoothed_df[key] = savgol_filter(df[key], window_size, 2, mode='nearest')
    return smoothed_df

class DirectImageGuide():
    """
    Image guide that uses an optimizer and torch autograd to optimize an image representation.
    Based on the BigGan+CLIP algorithm by advadnoun (https://twitter.com/advadnoun).
    """
    def __init__(self, image_rep, embedder, optimizer=None, lr=None, **optimizer_params):
        self.image_rep = image_rep
        self.embedder = embedder
        if lr is None:
            lr = image_rep.lr
        optimizer_params['lr'] = lr
        self.optimizer_params = optimizer_params
        if optimizer is None:
            self.optimizer = optim.Adam(image_rep.parameters(), **optimizer_params)
        else:
            self.optimizer = optimizer
        self.dataframe = []

    def run_steps(self, n_steps, prompts, interp_prompts, loss_augs, stop=-math.inf, interp_steps=0, i_offset=0, skipped_steps=0):
        """Runs the optimizer."""
        for i in tqdm(range(n_steps)):
            self.update(i + i_offset, i + skipped_steps)
            losses = self.train(i + skipped_steps, prompts, interp_prompts, loss_augs, interp_steps=interp_steps)
            if losses['TOTAL'] <= stop:
                break
        return i + 1

    def set_optim(self, opt=None):
        if opt is not None:
            self.optimizer = opt
        else:
            self.optimizer = optim.Adam(self.image_rep.parameters(), **self.optimizer_params)

    def clear_dataframe(self):
        self.dataframe = []

    def plot_losses(self, axs):
        def plot_dataframe(df, ax, legend=False):
            keys = df.columns.tolist()
            keys.sort(reverse=True, key=lambda k: df[k].iloc[-1])
            ax.clear()
            df[keys].plot(ax=ax, legend=legend)
            if legend:
                ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                           bottom=True, top=False, left=True, right=False)
            labelLines(ax.get_lines(), align=False)
            return

        dfs = self.dataframe[:]
        if dfs:
            dfs[0] = smooth_dataframe(dfs[0], 17)
        for i, (df, ax) in enumerate(zip(dfs, axs)):
            if len(df.index) < 2:
                return False
            if not df.empty:
                plot_dataframe(df, ax, legend=(i == 0))
            ax.set_ylabel('Loss')
            ax.set_xlabel('Step')
        return True

    def update(self, i, stage_i):
        """Update hook called every step."""
        pass

    def train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0, save_loss=True):
        """Performs a training step."""
        self.optimizer.zero_grad()
        z = self.image_rep.decode_training_tensor()

        # Precompute formatted inputs to avoid redundant computations
        if self.embedder is not None:
            image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)

            # Cache formatted inputs for prompts and interpolation prompts
            all_prompts = prompts + interp_prompts
            formatted_inputs = {}
            for prompt in set(all_prompts):
                formatted_inputs[prompt] = {
                    'embeds': format_input(image_embeds, self.embedder, prompt),
                    'offsets': format_input(offsets, self.embedder, prompt),
                    'sizes': format_input(sizes, self.embedder, prompt)
                }
        else:
            formatted_inputs = {}

        # Cache formatted inputs for loss augmentations
        formatted_z = {}
        for aug in loss_augs:
            formatted_z[aug] = format_input(z, self.image_rep, aug)

        # Compute interpolation factor
        if i < interp_steps:
            t = i / interp_steps
            interp_losses = [prompt(formatted_inputs[prompt]['embeds'],
                                    formatted_inputs[prompt]['offsets'],
                                    formatted_inputs[prompt]['sizes'])[0] * (1 - t) for prompt in interp_prompts]
        else:
            t = 1
            interp_losses = [0]

        # Compute prompt losses
        prompt_losses = {}
        for prompt in prompts:
            loss = prompt(formatted_inputs[prompt]['embeds'],
                          formatted_inputs[prompt]['offsets'],
                          formatted_inputs[prompt]['sizes'])
            # Scale loss by interpolation factor
            loss[0].mul_(t)
            prompt_losses[prompt] = loss

        # Compute augmentation losses
        aug_losses = {}
        for aug in loss_augs:
            loss = aug(formatted_z[aug], self.image_rep)
            aug_losses[aug] = loss

        # Compute image losses
        image_augs = self.image_rep.image_loss()
        image_losses = {}
        for aug in image_augs:
            loss = aug(self.image_rep)
            image_losses[aug] = loss

        # Aggregate losses
        total_loss = sum(loss[0] for loss in prompt_losses.values()) + \
                     sum(loss[0] for loss in aug_losses.values()) + \
                     sum(loss[0] for loss in image_losses.values()) + \
                     sum(interp_losses)

        # Prepare loss tracking
        if save_loss:
            loss_dict = {'TOTAL': float(total_loss)}
            loss_dict.update({str(k): float(v[0]) for k, v in prompt_losses.items()})
            loss_dict.update({str(k): float(v[0]) for k, v in aug_losses.items()})
            loss_dict.update({str(k): float(v[0]) for k, v in image_losses.items()})
            if not self.dataframe:
                self.dataframe = [pd.DataFrame(loss_dict, index=[i])]
            else:
                self.dataframe[0] = pd.concat([self.dataframe[0], pd.DataFrame(loss_dict, index=[i])])

        total_loss.backward()
        self.optimizer.step()
        self.image_rep.update()

        return {'TOTAL': float(total_loss)}

class SwarmImageGuide():
    """
    Image guide that uses Particle Swarm Optimization (PSO) instead of gradient descent.
    This approach uses a swarm of particles to explore the parameter space without gradients.
    """
    def __init__(self, image_rep, embedder, num_particles=30, c1=1.5, c2=1.5, w=0.7, **optimizer_params):
        self.image_rep = image_rep
        self.embedder = embedder
        
        # PSO hyperparameters
        self.num_particles = num_particles  # Number of particles in the swarm
        self.c1 = c1                        # Cognitive coefficient (personal best attraction)
        self.c2 = c2                        # Social coefficient (global best attraction)
        self.w = w                          # Inertia weight
        
        # Initialize the swarm
        self.particles = []
        self.velocities = []
        self.personal_bests = []
        self.personal_best_scores = []
        self.global_best = None
        self.global_best_score = float('inf')
        
        # Initialize particles with the current image parameters plus random variations
        self._initialize_swarm()
        
        self.dataframe = []
    
    def _initialize_swarm(self):
        """Initialize the swarm of particles."""
        # Get the current parameters of the image representation
        current_params = [p.data.clone() for p in self.image_rep.parameters()]
        
        # Create multiple variations of these parameters for our particles
        for i in range(self.num_particles):
            # Clone the current parameters for each particle
            particle_params = [p.clone() for p in current_params]
            
            # Add some random noise to create diversity
            for p in particle_params:
                noise = randn_like(p) * 0.01  # Small random variations
                p.add_(noise)
            
            self.particles.append(particle_params)
            
            # Initialize velocities to zero
            velocity = [zeros_like(p) for p in particle_params]
            self.velocities.append(velocity)
            
            # Initialize personal best to the starting position
            self.personal_bests.append([p.clone() for p in particle_params])
            self.personal_best_scores.append(float('inf'))
    
    def _evaluate_particle(self, particle_params, prompts, interp_prompts, loss_augs, i, interp_steps=0):
        """Evaluate the fitness of a particle (lower is better)."""
        # Temporarily set the image parameters to the particle's parameters
        original_params = [p.data.clone() for p in self.image_rep.parameters()]
        
        try:
            # Set image parameters to the particle's parameters
            for param, particle_param in zip(self.image_rep.parameters(), particle_params):
                param.data.copy_(particle_param)
            
            # Decode the image with the new parameters
            z = self.image_rep.decode_training_tensor()
            
            # Compute the loss (fitness) without computing gradients
            with torch.no_grad():
                # Similar to DirectImageGuide's train method but without gradient computation
                if self.embedder is not None:
                    image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)
                    
                    all_prompts = prompts + interp_prompts
                    formatted_inputs = {}
                    for prompt in set(all_prompts):
                        formatted_inputs[prompt] = {
                            'embeds': format_input(image_embeds, self.embedder, prompt),
                            'offsets': format_input(offsets, self.embedder, prompt),
                            'sizes': format_input(sizes, self.embedder, prompt)
                        }
                else:
                    formatted_inputs = {}
                
                formatted_z = {}
                for aug in loss_augs:
                    formatted_z[aug] = format_input(z, self.image_rep, aug)
                
                # Compute interpolation factor
                if i < interp_steps:
                    t = i / interp_steps
                    interp_losses = [prompt(formatted_inputs[prompt]['embeds'],
                                           formatted_inputs[prompt]['offsets'],
                                           formatted_inputs[prompt]['sizes'])[0] * (1 - t) for prompt in interp_prompts]
                else:
                    t = 1
                    interp_losses = [0]
                
                # Compute prompt losses
                prompt_losses = [prompt(formatted_inputs[prompt]['embeds'],
                                      formatted_inputs[prompt]['offsets'],
                                      formatted_inputs[prompt]['sizes'])[0] * t for prompt in prompts]
                
                # Compute augmentation losses
                aug_losses = [aug(formatted_z[aug], self.image_rep)[0] for aug in loss_augs]
                
                # Compute image losses
                image_augs = self.image_rep.image_loss()
                image_losses = [aug(self.image_rep)[0] for aug in image_augs]
                
                # Compute total loss
                total_loss = sum(prompt_losses) + sum(aug_losses) + sum(image_losses) + sum(interp_losses)
                
                return float(total_loss)
                
        finally:
            # Restore original parameters
            for param, original_param in zip(self.image_rep.parameters(), original_params):
                param.data.copy_(original_param)
    
    def update(self, i, stage_i):
        """Update hook called every step."""
        pass
    
    def train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0, save_loss=True):
        """Performs a training step using Particle Swarm Optimization."""
        losses = {}
        
        # Evaluate all particles
        for idx, particle in enumerate(self.particles):
            fitness = self._evaluate_particle(particle, prompts, interp_prompts, loss_augs, i, interp_steps)
            
            # Update personal best
            if fitness < self.personal_best_scores[idx]:
                self.personal_best_scores[idx] = fitness
                self.personal_bests[idx] = [p.clone() for p in particle]
                
                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best = [p.clone() for p in particle]
        
        # Update particle velocities and positions
        for idx in range(self.num_particles):
            for j, param in enumerate(self.particles[idx]):
                # Random coefficients
                r1 = rand_like(param)
                r2 = rand_like(param)
                
                # Update velocity
                cognitive_component = self.c1 * r1 * (self.personal_bests[idx][j] - param)
                social_component = self.c2 * r2 * (self.global_best[j] - param)
                
                self.velocities[idx][j] = self.w * self.velocities[idx][j] + cognitive_component + social_component
                
                # Update position
                self.particles[idx][j] = param + self.velocities[idx][j]
        
        # Apply the global best solution to the image
        for param, best_param in zip(self.image_rep.parameters(), self.global_best):
            param.data.copy_(best_param)
        
        # Update the image representation
        self.image_rep.update()
        
        # Track losses
        if save_loss:
            loss_dict = {'TOTAL': self.global_best_score}
            
            if not self.dataframe:
                self.dataframe = [pd.DataFrame(loss_dict, index=[i])]
            else:
                self.dataframe[0] = pd.concat([self.dataframe[0], pd.DataFrame(loss_dict, index=[i])])
        
        return {'TOTAL': self.global_best_score}
    
    def run_steps(self, n_steps, prompts, interp_prompts, loss_augs, stop=-math.inf, interp_steps=0, i_offset=0, skipped_steps=0):
        """Runs the optimizer."""
        for i in tqdm(range(n_steps)):
            self.update(i + i_offset, i + skipped_steps)
            losses = self.train(i + skipped_steps, prompts, interp_prompts, loss_augs, interp_steps=interp_steps)
            if losses['TOTAL'] <= stop:
                break
        return i + 1
    
    def clear_dataframe(self):
        self.dataframe = []
    
    def plot_losses(self, axs):
        """Plot the loss history."""
        def plot_dataframe(df, ax, legend=False):
            window_size = min(21, len(df) // 2)
            if window_size % 2 == 0:
                window_size -= 1
            if window_size < 3:
                window_size = 3
            
            smooth_df = smooth_dataframe(df, window_size)
            
            for key in smooth_df.columns:
                if key != 'TOTAL':
                    if float(smooth_df[key].iloc[-1]) != 0:
                        ax.plot(smooth_df.index, smooth_df[key], label=key)
            
            if legend:
                ax.legend()
        
        if self.dataframe:
            plot_dataframe(self.dataframe[0], axs[0], legend=False)
            ax2 = axs[0].twinx()
            ax2.plot(self.dataframe[0].index, self.dataframe[0]["TOTAL"], 'k--', label="TOTAL")
            ax2.set_ylabel("TOTAL Loss")
            labelLines(plt.gca().get_lines(), zorder=2.5)
