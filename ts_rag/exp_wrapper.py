import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class ExpLongTermForecastRAG(Exp_Long_Term_Forecast):
    def _output_slice(self):
        return slice(-1, None) if self.args.features == "MS" else slice(None)

    def _predict_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        model_x_mark = None if self.args.data == "PEMS" else batch_x_mark
        model_y_mark = None if self.args.data == "PEMS" else batch_y_mark

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, model_x_mark, dec_inp, model_y_mark)
        else:
            outputs = self.model(batch_x, model_x_mark, dec_inp, model_y_mark)

        output_slice = self._output_slice()
        outputs = outputs[:, -self.args.pred_len :, output_slice]
        batch_y_future = batch_y[:, -self.args.pred_len :, output_slice]
        return batch_x, batch_y_future, outputs

    def _make_eval_loader(self, flag):
        dataset, _ = self._get_data(flag=flag)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        return dataset, loader

    def load_checkpoint(self, setting, checkpoint_path=None):
        path = checkpoint_path or os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return path

    def smoke_train(self, setting, max_batches=1):
        _, loader = self._get_data(flag="train")
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        self.model.train()
        losses = []
        for batch_index, batch in enumerate(loader):
            optimizer.zero_grad()
            _, true, prediction = self._predict_batch(*batch)
            loss = criterion(prediction, true)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if batch_index + 1 >= max_batches:
                break
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "checkpoint.pth"))
        return float(np.mean(losses))

    def vali(self, vali_data, vali_loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                _, true, prediction = self._predict_batch(*batch)
                if self.args.data == "PEMS":
                    pred_np = prediction.detach().cpu().numpy()
                    true_np = true.detach().cpu().numpy()
                    pred_np = self._inverse_output(vali_data, pred_np)
                    true_np = self._inverse_output(vali_data, true_np)
                    losses.append(float(np.mean(np.abs(pred_np - true_np))) / 100.0)
                else:
                    losses.append(float(criterion(prediction, true).detach().cpu()))
        self.model.train()
        return float(np.mean(losses))

    @staticmethod
    def _inverse_output(dataset, values):
        shape = values.shape
        feature_count = dataset.scaler.n_features_in_
        if shape[-1] == feature_count:
            return dataset.inverse_transform(values.reshape(-1, feature_count)).reshape(shape)
        if shape[-1] == 1:
            return values * dataset.scaler.scale_[-1] + dataset.scaler.mean_[-1]
        raise ValueError(
            f"Cannot inverse {shape[-1]} output channels with a {feature_count}-feature scaler"
        )

    def predict_split(self, flag="test", inverse=False, limit=0):
        dataset, loader = self._make_eval_loader(flag)
        xs, xs_full, ys, preds, x_marks, y_marks = [], [], [], [], [], []
        store_full_separately = self.args.features == "MS"

        self.model.eval()
        seen = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x_scaled, batch_y_future_scaled, outputs_scaled = self._predict_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )

                x_np = batch_x_scaled.detach().cpu().numpy()
                y_np = batch_y_future_scaled.detach().cpu().numpy()
                pred_np = outputs_scaled.detach().cpu().numpy()

                xs.append(x_np[:, :, self._output_slice()])
                if store_full_separately:
                    xs_full.append(x_np)
                ys.append(y_np)
                preds.append(pred_np)
                x_marks.append(batch_x_mark.detach().cpu().numpy())
                y_marks.append(batch_y_mark[:, -self.args.pred_len :, :].detach().cpu().numpy())

                seen += x_np.shape[0]
                if limit and seen >= limit:
                    break

        x_scaled = np.concatenate(xs, axis=0)
        x_full_scaled = (
            np.concatenate(xs_full, axis=0) if store_full_separately else x_scaled
        )
        y_scaled = np.concatenate(ys, axis=0)
        pred_scaled = np.concatenate(preds, axis=0)
        x_mark = np.concatenate(x_marks, axis=0)
        y_mark = np.concatenate(y_marks, axis=0)
        if limit:
            x_scaled = x_scaled[:limit]
            x_full_scaled = x_full_scaled[:limit]
            y_scaled = y_scaled[:limit]
            pred_scaled = pred_scaled[:limit]
            x_mark = x_mark[:limit]
            y_mark = y_mark[:limit]

        bundle = {
            "split": flag,
            "x": x_scaled,
            "x_full": x_full_scaled,
            "x_mark": x_mark,
            "y": y_scaled,
            "y_mark": y_mark,
            "y_base": pred_scaled,
            "future_residual": y_scaled - pred_scaled,
            "output_scale": np.asarray(
                dataset.scaler.scale_[self._output_slice()], dtype=np.float64
            ),
            "output_mean": np.asarray(
                dataset.scaler.mean_[self._output_slice()], dtype=np.float64
            ),
        }

        if inverse and dataset.scale:
            bundle["x_inv"] = self._inverse_output(dataset, x_scaled)
            bundle["x_full_inv"] = (
                self._inverse_output(dataset, x_full_scaled)
                if store_full_separately
                else bundle["x_inv"]
            )
            bundle["y_inv"] = self._inverse_output(dataset, y_scaled)
            bundle["y_base_inv"] = self._inverse_output(dataset, pred_scaled)
            bundle["future_residual_inv"] = bundle["y_inv"] - bundle["y_base_inv"]

        return bundle
