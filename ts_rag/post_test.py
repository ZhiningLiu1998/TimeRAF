import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.timefeatures import time_features


class ETTPostTestDataset(Dataset):
    """The unused ETT timeline after the repository's official test split."""

    def __init__(self, args):
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.scale = True
        points_per_hour = 4 if args.dataset.startswith("ETTm") else 1
        train_end = 12 * 30 * 24 * points_per_hour
        official_end = 20 * 30 * 24 * points_per_hour

        path = os.path.join(args.root_path, args.data_path)
        frame = pd.read_csv(path)
        if args.features in {"M", "MS"}:
            values = frame.iloc[:, 1:].values
        else:
            values = frame[[args.target]].values

        self.scaler = StandardScaler().fit(values[:train_end])
        scaled = self.scaler.transform(values)
        start = official_end - self.seq_len
        self.data_x = scaled[start:]
        self.data_y = scaled[start:]

        dates = pd.to_datetime(frame["date"].iloc[start:].values)
        if args.embed == "timeF":
            self.data_stamp = time_features(dates, freq=args.freq).transpose(1, 0)
        else:
            stamp = pd.DataFrame({"date": dates})
            stamp["month"] = stamp.date.dt.month
            stamp["day"] = stamp.date.dt.day
            stamp["weekday"] = stamp.date.dt.weekday
            stamp["hour"] = stamp.date.dt.hour
            if points_per_hour == 4:
                stamp["minute"] = stamp.date.dt.minute // 15
            self.data_stamp = stamp.drop(columns=["date"]).values

        self.first_forecast_origin = official_end
        self.final_data_index = len(frame) - 1

    def __getitem__(self, index):
        input_start = index
        input_end = input_start + self.seq_len
        target_start = input_end - self.label_len
        target_end = target_start + self.label_len + self.pred_len
        return (
            self.data_x[input_start:input_end],
            self.data_y[target_start:target_end],
            self.data_stamp[input_start:input_end],
            self.data_stamp[target_start:target_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)


def predict_post_test(exp, args):
    dataset = ETTPostTestDataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    xs, ys, predictions, x_marks, y_marks = [], [], [], [], []

    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x, y, prediction = exp._predict_batch(
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            )
            xs.append(x.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            predictions.append(prediction.detach().cpu().numpy())
            x_marks.append(batch_x_mark.detach().cpu().numpy())
            y_marks.append(
                batch_y_mark[:, -args.pred_len :, :].detach().cpu().numpy()
            )

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    y_base = np.concatenate(predictions)
    return dataset, {
        "split": "post_test",
        "x": x,
        "x_mark": np.concatenate(x_marks),
        "y": y,
        "y_mark": np.concatenate(y_marks),
        "y_base": y_base,
        "future_residual": y - y_base,
    }
