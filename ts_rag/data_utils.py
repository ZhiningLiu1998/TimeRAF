from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider


def load_window_bundle(args, split="test", max_samples=3):
    dataset, _ = data_provider(args, split)
    loader = DataLoader(
        dataset,
        batch_size=max_samples,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    batch_x, batch_y, _, _ = next(iter(loader))
    return {
        "split": split,
        "x": batch_x.numpy(),
        "y": batch_y[:, -args.pred_len :, :].numpy(),
    }
