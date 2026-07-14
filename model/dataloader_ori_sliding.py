import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader


class GenRecDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False):
        super(GenRecDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch, pad_token=0):
        histories = [item["history"] for item in batch]
        targets = [item["target"] for item in batch]

        flattened_histories = torch.stack(
            [torch.tensor([elem for code in history for elem in code], dtype=torch.long) for history in histories]
        )
        flattened_targets = torch.stack([torch.tensor(target, dtype=torch.long) for target in targets])
        attention_masks = (flattened_histories != pad_token).long()

        if "history_text_emb" not in batch[0]:
            return {
                "history": flattened_histories,
                "target": flattened_targets,
                "attention_mask": attention_masks,
            }

        text_emb_tensors = []
        for item in batch:
            emb = item["history_text_emb"]
            if isinstance(emb, np.ndarray):
                if emb.dtype == object:
                    emb = [e.tolist() if isinstance(e, np.ndarray) else e for e in emb]
                else:
                    emb = emb.tolist()
            if not isinstance(emb, list):
                emb = [emb] if emb is not None else []
            text_emb_tensors.append(torch.tensor(emb, dtype=torch.float32))

        padded_text_embs = rnn_utils.pad_sequence(text_emb_tensors, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([len(tensor) for tensor in text_emb_tensors])
        max_len = padded_text_embs.size(1)
        encoder_attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()

        return {
            "history": flattened_histories,
            "target": flattened_targets,
            "attention_mask": attention_masks,
            "history_text_emb": padded_text_embs,
            "encoder_attention_mask": encoder_attention_mask,
        }
