import argparse
import os
import torch
import torch.nn as nn
import joblib
import numpy as np

from src.m3_modeling.data_loader import prepare_multi_horizon_data, prepare_walk_forward_folds
from src.m3_modeling.lstm_model import MultiHorizonLSTM, mc_dropout_inference
from src.m3_modeling.baseline_lstm import BaselineLSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_single_model(model, train_loader, val_loader, epochs, model_name, save_path,
                       patience=7, lr=0.001, lr_patience=3, lr_factor=0.5):
    """
    Train a model with early stopping, LR scheduling, and gradient clipping.
    Returns best validation loss achieved.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )

    best_loss = float('inf')
    patience_counter = 0

    print(f"\n--- Training {model_name} (max {epochs} epochs, patience={patience}) ---")
    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                y_pred = model(batch_x)
                val_loss += criterion(y_pred, batch_y).item()

        avg_val = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch + 1}/{epochs} | Train: {avg_train:.5f} | Val: {avg_val:.5f} | LR: {current_lr:.6f}")

        # LR scheduler step
        scheduler.step(avg_val)

        # Early stopping check
        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    print(f"  Best {model_name} val loss: {best_loss:.5f} → {save_path}")
    return best_loss


def evaluate_model(model, x_test, y_test, t_scaler, model_name):
    """Evaluate model with MC Dropout and report per-horizon metrics."""
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32)

    # MC Dropout inference
    mean_pred, std_pred = mc_dropout_inference(model, x_tensor, num_passes=30)

    # Inverse transform
    mean_real = t_scaler.inverse_transform(mean_pred)
    y_real = t_scaler.inverse_transform(y_test) if hasattr(y_test, 'shape') else y_test

    horizons = ["1D", "5D", "21D", "126D"]
    print(f"\n--- {model_name} Evaluation ---")
    for i, h in enumerate(horizons):
        mae = mean_absolute_error(y_real[:, i], mean_real[:, i])
        dir_acc = np.sum(np.sign(mean_real[:, i]) == np.sign(y_real[:, i])) / len(y_real) * 100
        avg_std = np.mean(std_pred[:, i])
        print(f"  [{h}] MAE: {mae:.5f} | Dir Acc: {dir_acc:.2f}% | Avg Uncertainty: {avg_std:.5f}")


def main():
    parser = argparse.ArgumentParser(description="M3 Model Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory containing Parquet files (default: data/processed)")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs per model")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Primary LSTM hidden dimensions")
    parser.add_argument("--layers", type=int, default=2, help="Primary LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--walk_forward", action="store_true",
                        help="Run walk-forward cross-validation before final training")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of walk-forward folds")

    args = parser.parse_args()

    # Ensure artifacts directory exists
    model_dir = "artifacts/models"
    os.makedirs(model_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  M3 Training Pipeline — Real Data Mode")
    print(f"  Data: {args.data_dir}")
    print(f"  Epochs: {args.epochs} | Patience: {args.patience}")
    print(f"{'=' * 60}")

    # ══════════════════════════════════════════════════════════════
    # OPTIONAL: Walk-Forward Cross-Validation (Issue #5)
    # ══════════════════════════════════════════════════════════════
    if args.walk_forward:
        print(f"\n{'=' * 60}")
        print(f"  Walk-Forward Cross-Validation ({args.n_folds} folds)")
        print(f"{'=' * 60}")

        folds = prepare_walk_forward_folds(args.data_dir, n_folds=args.n_folds)
        fold_metrics = []

        for fold in folds:
            fold_idx = fold['fold_idx']
            num_features = fold['num_features']
            print(f"\n{'─' * 40}")
            print(f"  Fold {fold_idx + 1}/{len(folds)} (train={fold['train_size']}, val={fold['val_size']})")
            print(f"{'─' * 40}")

            model = MultiHorizonLSTM(
                input_dim=num_features, hidden_dim=args.hidden_dim,
                num_layers=args.layers, dropout_rate=args.dropout, output_dim=4
            )

            fold_path = os.path.join(model_dir, f"fold_{fold_idx}_lstm.pth")
            best_loss = train_single_model(
                model, fold['train_loader'], fold['val_loader'],
                epochs=args.epochs, model_name=f"Fold-{fold_idx + 1} LSTM",
                save_path=fold_path, patience=args.patience
            )

            # Load best weights and evaluate
            model.load_state_dict(torch.load(fold_path, map_location='cpu', weights_only=True))

            # Scale val data for evaluation
            y_val_scaled = fold['target_scaler'].transform(fold['y_val_raw'])
            evaluate_model(model, fold['x_val_raw'], fold['y_val_raw'],
                           fold['target_scaler'], f"Fold-{fold_idx + 1}")

            # Compute directional accuracy for 21D
            x_tensor = torch.tensor(
                fold['feature_scaler'].transform(
                    fold['x_val_raw'].reshape(-1, fold['x_val_raw'].shape[2])
                ).reshape(fold['x_val_raw'].shape),
                dtype=torch.float32
            )
            mean_pred, _ = mc_dropout_inference(model, x_tensor, num_passes=20)
            mean_real = fold['target_scaler'].inverse_transform(mean_pred)
            dir_acc_21d = np.sum(
                np.sign(mean_real[:, 2]) == np.sign(fold['y_val_raw'][:, 2])
            ) / len(fold['y_val_raw']) * 100

            fold_metrics.append({
                'fold': fold_idx + 1,
                'val_loss': best_loss,
                'dir_acc_21d': dir_acc_21d,
            })

            # Clean up fold checkpoint
            if os.path.exists(fold_path):
                os.remove(fold_path)

        print(f"\n{'=' * 60}")
        print(f"  Walk-Forward Summary")
        print(f"{'=' * 60}")
        for fm in fold_metrics:
            print(f"  Fold {fm['fold']}: Val Loss={fm['val_loss']:.5f} | 21D Dir Acc={fm['dir_acc_21d']:.2f}%")
        avg_dir = np.mean([fm['dir_acc_21d'] for fm in fold_metrics])
        print(f"  Average 21D Directional Accuracy: {avg_dir:.2f}%")
        print()

    # ══════════════════════════════════════════════════════════════
    # FINAL TRAINING: Train on full train split (80/20)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Final Production Training")
    print(f"{'=' * 60}")

    # 1. Load Data
    try:
        train_loader, test_loader, f_scaler, t_scaler, x_test_np, y_test_np, num_features = \
            prepare_multi_horizon_data(args.data_dir)
    except Exception as e:
        print(f"Data Loading Error: {e}")
        return

    # Save Scalers
    joblib.dump(f_scaler, os.path.join(model_dir, "feature_scaler.pkl"))
    joblib.dump(t_scaler, os.path.join(model_dir, "target_scaler.pkl"))
    print(f"Saved scalers to {model_dir}")

    # ── Train Primary LSTM (2-layer, hidden=64) ──────────────────────
    primary_model = MultiHorizonLSTM(
        input_dim=num_features, hidden_dim=args.hidden_dim,
        num_layers=args.layers, dropout_rate=args.dropout, output_dim=4
    )
    primary_path = os.path.join(model_dir, "best_lstm_primary.pth")

    train_single_model(
        primary_model, train_loader, test_loader,
        epochs=args.epochs, model_name="Primary LSTM (2L-64H)",
        save_path=primary_path, patience=args.patience
    )

    # Load best weights and evaluate
    primary_model.load_state_dict(torch.load(primary_path, map_location='cpu', weights_only=True))

    # Scale x_test for evaluation
    n_test, seq_len, n_feat = x_test_np.shape
    x_test_scaled = f_scaler.transform(x_test_np.reshape(-1, n_feat)).reshape(n_test, seq_len, n_feat)
    evaluate_model(primary_model, x_test_scaled, y_test_np, t_scaler, "Primary LSTM")

    # ── Train Secondary LSTM (1-layer, hidden=128) — True Ensemble ───
    secondary_model = BaselineLSTM(
        input_dim=num_features, hidden_dim=128,
        num_layers=1, output_dim=4, dropout_rate=0.2
    )
    secondary_path = os.path.join(model_dir, "best_lstm_secondary.pth")

    train_single_model(
        secondary_model, train_loader, test_loader,
        epochs=args.epochs, model_name="Secondary LSTM (1L-128H)",
        save_path=secondary_path, patience=args.patience
    )

    # Load best weights and evaluate
    secondary_model.load_state_dict(torch.load(secondary_path, map_location='cpu', weights_only=True))
    evaluate_model(secondary_model, x_test_scaled, y_test_np, t_scaler, "Secondary LSTM")

    # ── Also save as legacy path for backward compatibility ──
    # Copy primary as best_lstm.pth
    import shutil
    shutil.copy2(primary_path, os.path.join(model_dir, "best_lstm.pth"))

    print(f"\n{'=' * 60}")
    print(f"  Pipeline Complete")
    print(f"  Primary model:   {primary_path}")
    print(f"  Secondary model: {secondary_path}")
    print(f"  Scalers:         {model_dir}/feature_scaler.pkl, target_scaler.pkl")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
