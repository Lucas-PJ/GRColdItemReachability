import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


class OPQTokenizer:
    def __init__(
        self,
        in_dim: int,
        num_emb_list: List[int],
        proj_dim: int = 0,
        random_state: int = 2024,
        center_data: bool = True,
        kmeans_n_init: int = 10,
    ):
        self.in_dim = int(in_dim)
        self.num_emb_list = [int(x) for x in num_emb_list]
        self.num_codebooks = len(self.num_emb_list)
        self.random_state = int(random_state)
        self.center_data = bool(center_data)
        self.kmeans_n_init = int(kmeans_n_init)
        self.proj_dim = int(proj_dim) if int(proj_dim) > 0 else int(in_dim)

        if self.proj_dim > self.in_dim:
            raise ValueError(f"proj_dim={self.proj_dim} cannot exceed in_dim={self.in_dim}")
        if self.proj_dim % self.num_codebooks != 0:
            raise ValueError(
                f"proj_dim={self.proj_dim} must be divisible by num_codebooks={self.num_codebooks}"
            )

        self.subvector_dim = self.proj_dim // self.num_codebooks
        self.mean_: Optional[np.ndarray] = None
        self.proj_matrix_: Optional[np.ndarray] = None
        self.rotation_: Optional[np.ndarray] = None
        self.codebooks_: Optional[List[np.ndarray]] = None
        self.history_: List[Dict[str, float]] = []

    def state_dict(self) -> Dict[str, object]:
        return {
            "in_dim": self.in_dim,
            "proj_dim": self.proj_dim,
            "num_emb_list": self.num_emb_list,
            "num_codebooks": self.num_codebooks,
            "subvector_dim": self.subvector_dim,
            "random_state": self.random_state,
            "center_data": self.center_data,
            "kmeans_n_init": self.kmeans_n_init,
            "mean_": self.mean_,
            "proj_matrix_": self.proj_matrix_,
            "rotation_": self.rotation_,
            "codebooks_": self.codebooks_,
            "history_": self.history_,
        }

    def load_state_dict(self, state_dict: Dict[str, object]):
        self.in_dim = int(state_dict["in_dim"])
        self.proj_dim = int(state_dict["proj_dim"])
        self.num_emb_list = [int(x) for x in state_dict["num_emb_list"]]
        self.num_codebooks = int(state_dict["num_codebooks"])
        self.subvector_dim = int(state_dict["subvector_dim"])
        self.random_state = int(state_dict.get("random_state", 2024))
        self.center_data = bool(state_dict.get("center_data", True))
        self.kmeans_n_init = int(state_dict.get("kmeans_n_init", 10))
        self.mean_ = None if state_dict.get("mean_") is None else np.asarray(state_dict["mean_"], dtype=np.float32)
        self.proj_matrix_ = np.asarray(state_dict["proj_matrix_"], dtype=np.float32)
        self.rotation_ = np.asarray(state_dict["rotation_"], dtype=np.float32)
        self.codebooks_ = [np.asarray(cb, dtype=np.float32) for cb in state_dict["codebooks_"]]
        self.history_ = list(state_dict.get("history_", []))
        return self

    def _check_is_fitted(self):
        if self.rotation_ is None or self.codebooks_ is None or self.proj_matrix_ is None:
            raise RuntimeError("OPQTokenizer is not fitted yet.")

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected input shape [N, {self.in_dim}], got {x.shape}")
        return x

    def _center(self, x: np.ndarray) -> np.ndarray:
        if self.center_data:
            return x - self.mean_
        return x

    def _project(self, x: np.ndarray) -> np.ndarray:
        return self._center(x) @ self.proj_matrix_

    def _rotate(self, x_proj: np.ndarray) -> np.ndarray:
        return x_proj @ self.rotation_

    def _split_subspaces(self, z: np.ndarray) -> List[np.ndarray]:
        return [
            z[:, idx * self.subvector_dim : (idx + 1) * self.subvector_dim]
            for idx in range(self.num_codebooks)
        ]

    def _fit_projection(self, x: np.ndarray) -> np.ndarray:
        if self.center_data:
            self.mean_ = x.mean(axis=0, keepdims=True).astype(np.float32)
            x_centered = x - self.mean_
        else:
            self.mean_ = np.zeros((1, self.in_dim), dtype=np.float32)
            x_centered = x

        if self.proj_dim == self.in_dim:
            self.proj_matrix_ = np.eye(self.in_dim, dtype=np.float32)
            return x_centered

        _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
        self.proj_matrix_ = vt[: self.proj_dim].T.astype(np.float32)
        return x_centered @ self.proj_matrix_

    def _update_codebooks(
        self,
        z_rot: np.ndarray,
        kmeans_iters: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        labels_all = []
        recons = []
        codebooks = []
        for subspace_idx, (k, subspace) in enumerate(zip(self.num_emb_list, self._split_subspaces(z_rot))):
            kmeans = KMeans(
                n_clusters=k,
                max_iter=kmeans_iters,
                n_init=self.kmeans_n_init,
                random_state=self.random_state + subspace_idx,
            )
            labels = kmeans.fit_predict(subspace)
            centers = kmeans.cluster_centers_.astype(np.float32)
            recon = centers[labels]
            labels_all.append(labels.astype(np.int64))
            recons.append(recon)
            codebooks.append(centers)
        z_hat = np.concatenate(recons, axis=1)
        labels_matrix = np.stack(labels_all, axis=1)
        return z_hat, labels_matrix, codebooks

    @staticmethod
    def _orthogonal_procrustes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = x.T @ y
        u, _, vt = np.linalg.svd(m, full_matrices=False)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        return r.astype(np.float32)

    @staticmethod
    def collision_rate(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        keys = [tuple(int(x) for x in row) for row in indices]
        unique = len(set(keys))
        return float((len(keys) - unique) / len(keys))

    def fit(
        self,
        x: np.ndarray,
        opq_iters: int = 30,
        kmeans_iters: int = 50,
        tol: float = 1e-6,
        verbose: bool = True,
    ):
        x = self._prepare_input(x)
        x_proj = self._fit_projection(x)
        self.rotation_ = np.eye(self.proj_dim, dtype=np.float32)
        best_loss = math.inf
        best_state = None
        prev_loss = None
        self.history_ = []

        for iteration in range(int(opq_iters)):
            z_rot = self._rotate(x_proj)
            z_hat, labels, codebooks = self._update_codebooks(z_rot, kmeans_iters=int(kmeans_iters))
            self.codebooks_ = codebooks
            self.rotation_ = self._orthogonal_procrustes(x_proj, z_hat)
            z_rot = self._rotate(x_proj)
            z_hat, labels, codebooks = self._update_codebooks(z_rot, kmeans_iters=max(10, int(kmeans_iters // 2)))
            self.codebooks_ = codebooks

            loss = float(np.mean((z_rot - z_hat) ** 2))
            collision = self.collision_rate(labels)
            stats = {
                "iteration": int(iteration),
                "loss": loss,
                "collision_rate": collision,
            }
            self.history_.append(stats)
            if verbose:
                print(f"[OPQ] iter={iteration:03d} loss={loss:.8f} collision={collision:.8f}")

            if loss < best_loss:
                best_loss = loss
                best_state = {
                    "rotation_": self.rotation_.copy(),
                    "codebooks_": [cb.copy() for cb in self.codebooks_],
                }

            if prev_loss is not None and abs(prev_loss - loss) < tol:
                if verbose:
                    print(f"[OPQ] early stop at iter={iteration} due to tolerance {tol}")
                break
            prev_loss = loss

        if best_state is not None:
            self.rotation_ = best_state["rotation_"]
            self.codebooks_ = best_state["codebooks_"]
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x = self._prepare_input(x)
        z_rot = self._rotate(self._project(x))
        labels_all = []
        for centers, subspace in zip(self.codebooks_, self._split_subspaces(z_rot)):
            distances = (
                np.sum(subspace ** 2, axis=1, keepdims=True)
                + np.sum(centers ** 2, axis=1)[None, :]
                - 2.0 * subspace @ centers.T
            )
            labels = np.argmin(distances, axis=1).astype(np.int64)
            labels_all.append(labels)
        return np.stack(labels_all, axis=1)

    def reconstruct(self, indices: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        indices = np.asarray(indices, dtype=np.int64)
        parts = []
        for subspace_idx, centers in enumerate(self.codebooks_):
            parts.append(centers[indices[:, subspace_idx]])
        z_hat = np.concatenate(parts, axis=1)
        x_proj_hat = z_hat @ self.rotation_.T
        x_hat = x_proj_hat @ self.proj_matrix_.T
        if self.center_data:
            x_hat = x_hat + self.mean_
        return x_hat.astype(np.float32)

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        self.fit(x, **kwargs)
        return self.encode(x)
