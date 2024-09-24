from torch import optim, nn
from pytti.Notebook import tqdm
from pytti import *
import pandas as pd
import math

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
