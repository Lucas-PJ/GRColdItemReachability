import logging
import os
from time import time

import numpy as np
import torch

from utils import ensure_dir, get_local_time, set_color


class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        self.device = args.device
        self.ckpt_dir = os.path.join(f"./ckpt/{args.dataset}", get_local_time())
        ensure_dir(self.ckpt_dir)
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"

    def _save_checkpoint(self, iteration, ckpt_file, loss, collision_rate):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file)
        state = {
            "args": self.args,
            "iteration": iteration,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "loss": loss,
            "collision_rate": collision_rate,
        }
        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")
        return ckpt_path

    def fit(self, data):
        if hasattr(data, "dataset") and hasattr(data.dataset, "embeddings"):
            embeddings = data.dataset.embeddings
        elif hasattr(data, "embeddings"):
            embeddings = data.embeddings
        else:
            embeddings = []
            for batch in data:
                embeddings.append(batch.numpy())
            embeddings = np.concatenate(embeddings, axis=0)

        start = time()
        self.model.fit(
            embeddings,
            opq_iters=self.args.opq_iters,
            kmeans_iters=self.args.kmeans_iters,
            tol=self.args.tol,
            verbose=True,
        )
        end = time()

        history = self.model.history_
        if not history:
            raise RuntimeError("OPQ training produced empty history.")

        for stats in history:
            loss = float(stats["loss"])
            collision_rate = float(stats["collision_rate"])
            iteration = int(stats["iteration"])
            self.logger.info(
                set_color(f"iter {iteration} training", "green")
                + f" [time: {end - start:.2f}s, train loss: {loss:.8f}, collision_rate: {collision_rate:.8f}]"
            )
            if loss < self.best_loss:
                self.best_loss = loss
                self._save_checkpoint(iteration, self.best_loss_ckpt, loss, collision_rate)
            if collision_rate < self.best_collision_rate:
                self.best_collision_rate = collision_rate
                self._save_checkpoint(iteration, self.best_collision_ckpt, loss, collision_rate)

        final_stats = history[-1]
        self._save_checkpoint(
            int(final_stats["iteration"]),
            f"epoch_{int(final_stats['iteration'])}_collision_{float(final_stats['collision_rate']):.4f}_model.pth",
            float(final_stats["loss"]),
            float(final_stats["collision_rate"]),
        )
        return self.best_loss, self.best_collision_rate
