import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, Any, Tuple
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, accuracy_score
import matplotlib.pyplot as plt

class MatchScoreNeuralNetwork(nnx.Module):
    
    def __init__(self, rngs: nnx.Rngs, input_dim: int = 28):
        self.feature_dense1 = nnx.Linear(input_dim, 64, rngs=rngs)
        self.feature_dense2 = nnx.Linear(64, 32, rngs=rngs)
        
        self.red_pathway1 = nnx.Linear(32, 48, rngs=rngs)
        self.red_pathway2 = nnx.Linear(48, 24, rngs=rngs)
        
        self.blue_pathway1 = nnx.Linear(32, 48, rngs=rngs)
        self.blue_pathway2 = nnx.Linear(48, 24, rngs=rngs)
        
        self.interaction = nnx.Linear(48, 32, rngs=rngs)  
        
        self.final_dense1 = nnx.Linear(32, 16, rngs=rngs)
        self.final_dense2 = nnx.Linear(16, 8, rngs=rngs)
        
        self.red_output = nnx.Linear(8, 1, rngs=rngs)
        self.blue_output = nnx.Linear(8, 1, rngs=rngs)
        
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        self.bn2 = nnx.BatchNorm(32, rngs=rngs)
        self.bn_red1 = nnx.BatchNorm(48, rngs=rngs)
        self.bn_red2 = nnx.BatchNorm(24, rngs=rngs)
        self.bn_blue1 = nnx.BatchNorm(48, rngs=rngs)
        self.bn_blue2 = nnx.BatchNorm(24, rngs=rngs)
        self.bn_interaction = nnx.BatchNorm(32, rngs=rngs)
        self.bn_final1 = nnx.BatchNorm(16, rngs=rngs)
        self.bn_final2 = nnx.BatchNorm(8, rngs=rngs)
        
        self.dropout1 = nnx.Dropout(0.1, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.15, rngs=rngs)
        self.dropout_red = nnx.Dropout(0.1, rngs=rngs)
        self.dropout_blue = nnx.Dropout(0.1, rngs=rngs)
        self.dropout_interaction = nnx.Dropout(0.2, rngs=rngs)
        self.dropout_final = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x, training: bool = False):
        # Feature engineering
        x = self.feature_dense1(x)
        x = self.bn1(x, use_running_average=not training)
        x = jax.nn.swish(x)  
        x = self.dropout1(x, deterministic=not training)
        
        x = self.feature_dense2(x)
        x = self.bn2(x, use_running_average=not training)
        x = jax.nn.swish(x)
        x = self.dropout2(x, deterministic=not training)
        
        red_features = self.red_pathway1(x)
        red_features = self.bn_red1(red_features, use_running_average=not training)
        red_features = jax.nn.swish(red_features)
        red_features = self.dropout_red(red_features, deterministic=not training)
        
        red_features = self.red_pathway2(red_features)
        red_features = self.bn_red2(red_features, use_running_average=not training)
        red_features = jax.nn.swish(red_features)
        
        # Blue team pathway
        blue_features = self.blue_pathway1(x)
        blue_features = self.bn_blue1(blue_features, use_running_average=not training)
        blue_features = jax.nn.swish(blue_features)
        blue_features = self.dropout_blue(blue_features, deterministic=not training)
        
        blue_features = self.blue_pathway2(blue_features)
        blue_features = self.bn_blue2(blue_features, use_running_average=not training)
        blue_features = jax.nn.swish(blue_features)
        
        # Interaction layer
        combined = jnp.concatenate([red_features, blue_features], axis=-1)
        interaction = self.interaction(combined)
        interaction = self.bn_interaction(interaction, use_running_average=not training)
        interaction = jax.nn.swish(interaction)
        interaction = self.dropout_interaction(interaction, deterministic=not training)
        
        final = self.final_dense1(interaction)
        final = self.bn_final1(final, use_running_average=not training)
        final = jax.nn.swish(final)
        final = self.dropout_final(final, deterministic=not training)
        
        final = self.final_dense2(final)
        final = self.bn_final2(final, use_running_average=not training)
        final = jax.nn.swish(final)
        
        red_score = self.red_output(final)
        blue_score = self.blue_output(final)
        
        red_score = jax.nn.softplus(red_score)
        blue_score = jax.nn.softplus(blue_score)
        
        return jnp.concatenate([red_score, blue_score], axis=-1)


class MatchScoreTrainer:
    
    def __init__(self, model: MatchScoreNeuralNetwork, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=0.01))
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 
            'train_mse': [], 'val_mse': [], 'train_win_acc': [], 'val_win_acc': []
        }
        
        self.scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=1000,
            decay_rate=0.95
        )

    @staticmethod
    @nnx.jit
    def huber_loss(predictions, targets, delta=1.0):
        predictions = jnp.array(predictions)
        targets = jnp.array(targets)
        residual = jnp.abs(predictions - targets)
        condition = residual <= delta
        squared_loss = 0.5 * (residual ** 2)
        linear_loss = delta * residual - 0.5 * (delta ** 2)
        return jnp.mean(jnp.where(condition, squared_loss, linear_loss))
    
    @staticmethod
    @nnx.jit
    def combined_loss(predictions, targets, alpha=0.7):
        huber = MatchScoreTrainer.huber_loss(predictions, targets)
        mse = jnp.mean((predictions - targets) ** 2)
        return alpha * huber + (1 - alpha) * mse

    @staticmethod
    @nnx.jit
    def compute_metrics(predictions, targets):
        predictions = jnp.array(predictions)
        targets = jnp.array(targets)
        
        mae = jnp.mean(jnp.abs(predictions - targets))
        mse = jnp.mean((predictions - targets) ** 2)
        rmse = jnp.sqrt(mse)
        
        pred_wins = predictions[:, 1] > predictions[:, 0]  # Blue wins if blue > red
        true_wins = targets[:, 1] > targets[:, 0]
        win_accuracy = jnp.mean(pred_wins == true_wins)
        
        red_mae = jnp.mean(jnp.abs(predictions[:, 0] - targets[:, 0]))
        blue_mae = jnp.mean(jnp.abs(predictions[:, 1] - targets[:, 1]))
        red_mse = jnp.mean((predictions[:, 0] - targets[:, 0]) ** 2)
        blue_mse = jnp.mean((predictions[:, 1] - targets[:, 1]) ** 2)
        
        return {
            'mae': mae, 'mse': mse, 'rmse': rmse, 'win_accuracy': win_accuracy,
            'red_mae': red_mae, 'blue_mae': blue_mae,
            'red_mse': red_mse, 'blue_mse': blue_mse
        }

    @staticmethod
    @nnx.jit
    def train_step(model: MatchScoreNeuralNetwork, batch_x: jnp.ndarray, batch_y: jnp.ndarray, optimizer: nnx.Optimizer):
        def loss_fn(model: MatchScoreNeuralNetwork):
            predictions = model(batch_x, training=True)
            loss = MatchScoreTrainer.combined_loss(predictions, batch_y)
            return loss
        
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model)
        optimizer.update(grads)
        
        predictions = model(batch_x, training=True)
        metrics = MatchScoreTrainer.compute_metrics(predictions, batch_y)
        return loss, metrics

    @staticmethod
    @nnx.jit
    def val_step(model, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        predictions = model(batch_x, training=False)
        loss = MatchScoreTrainer.combined_loss(predictions, batch_y)
        metrics = MatchScoreTrainer.compute_metrics(predictions, batch_y)
        return loss, metrics
    
    def create_batches(self, X, y, batch_size=32, shuffle=True):
        n_samples = X.shape[0]
        indices = jnp.arange(n_samples)
        
        if shuffle:
            key = jax.random.PRNGKey(np.random.randint(0, 10000))
            indices = jax.random.permutation(key, indices)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 10,
              batch_size: int = 32,
              early_stop: int = 5,
              min_delta: float = 0.001) -> Dict[str, Any]:

        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train).reshape(-1, 2)
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val).reshape(-1, 2)
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        total_batches = (len(X_train) + batch_size - 1) // batch_size
        print(f"Starting training for {epochs} epochs...")
        print(f"Total batches per epoch: {total_batches}")
        print("-" * 80)
        
        for epoch in range(epochs):
            train_losses = []
            train_maes = []
            train_mses = []
            train_win_accs = []
            
            batch_idx = 0
            for batch_x, batch_y in self.create_batches(X_train, y_train, batch_size):
                loss, metrics = MatchScoreTrainer.train_step(self.model, batch_x, batch_y, self.optimizer)
                train_losses.append(loss)
                train_maes.append(metrics['mae'])
                train_mses.append(metrics['mse'])
                train_win_accs.append(metrics['win_accuracy'])
                
                if batch_idx % 50 == 0 and batch_idx > 0:
                    current_loss = jnp.mean(jnp.array(train_losses[-50:]))
                    current_mae = jnp.mean(jnp.array(train_maes[-50:]))
                    current_acc = jnp.mean(jnp.array(train_win_accs[-50:]))
                    print(f"  Epoch {epoch:3d}, Batch {batch_idx:4d}: "
                          f"Loss: {current_loss:.4f}, MAE: {current_mae:.4f}, Win Acc: {current_acc:.3f}")
                
                batch_idx += 1
            
            val_losses = []
            val_maes = []
            val_mses = []
            val_win_accs = []
            
            for batch_x, batch_y in self.create_batches(X_val, y_val, batch_size, shuffle=False):
                loss, metrics = MatchScoreTrainer.val_step(self.model, batch_x, batch_y)
                val_losses.append(loss)
                val_maes.append(metrics['mae'])
                val_mses.append(metrics['mse'])
                val_win_accs.append(metrics['win_accuracy'])
            
            epoch_train_loss = jnp.mean(jnp.array(train_losses))
            epoch_train_mae = jnp.mean(jnp.array(train_maes))
            epoch_train_mse = jnp.mean(jnp.array(train_mses))
            epoch_train_win_acc = jnp.mean(jnp.array(train_win_accs))
            
            epoch_val_loss = jnp.mean(jnp.array(val_losses))
            epoch_val_mae = jnp.mean(jnp.array(val_maes))
            epoch_val_mse = jnp.mean(jnp.array(val_mses))
            epoch_val_win_acc = jnp.mean(jnp.array(val_win_accs))
            
            self.history['train_loss'].append(float(epoch_train_loss))
            self.history['val_loss'].append(float(epoch_val_loss))
            self.history['train_mae'].append(float(epoch_train_mae))
            self.history['val_mae'].append(float(epoch_val_mae))
            self.history['train_mse'].append(float(epoch_train_mse))
            self.history['val_mse'].append(float(epoch_val_mse))
            self.history['train_win_acc'].append(float(epoch_train_win_acc))
            self.history['val_win_acc'].append(float(epoch_val_win_acc))
            
            print(f"EPOCH {epoch:3d} SUMMARY:")
            print(f"  Train - Loss: {epoch_train_loss:.4f}, MAE: {epoch_train_mae:.4f}, MSE: {epoch_train_mse:.4f}, Win Acc: {epoch_train_win_acc:.3f}")
            print(f"  Val   - Loss: {epoch_val_loss:.4f}, MAE: {epoch_val_mae:.4f}, MSE: {epoch_val_mse:.4f}, Win Acc: {epoch_val_win_acc:.3f}")
            
            if epoch_val_loss < best_val_loss - min_delta:
                best_val_loss = epoch_val_loss
                early_stop_counter = 0
                self.best_model_state = nnx.state(self.model)
                print(f"  *** New best val loss: {best_val_loss:.4f} ***")
            else:
                early_stop_counter += 1
                print(f"  early_stop: {early_stop_counter}/{early_stop}")
                if early_stop_counter >= early_stop:
                    print(f"  Early stopping at epoch {epoch}")
                    nnx.update(self.model, self.best_model_state)
                    break
            
            print("-" * 80)
        
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = jnp.array(X)
        predictions = self.model(X, training=False)
        return np.array(predictions)


class MatchScoreDataSetup:
    def __init__(self, use_robust_scaler=True):
        # Use RobustScaler for better handling of outliers
        self.feature_scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names = [
            'comps_attended_red_1', 'comps_attended_red_2', 'comps_attended_blue_1', 'comps_attended_blue_2',
            'trueskill_red_1', 'trueskill_red_2', 'trueskill_blue_1', 'trueskill_blue_2',
            'opr_red_1', 'opr_red_2', 'opr_blue_1', 'opr_blue_2',
            'dpr_red_1', 'dpr_red_2', 'dpr_blue_1', 'dpr_blue_2',
            'awp_per_match_red_1', 'awp_per_match_red_2', 'awp_per_match_blue_1', 'awp_per_match_blue_2',
            'ap_per_match_red_1', 'ap_per_match_red_2', 'ap_per_match_blue_1', 'ap_per_match_blue_2',
            'wp_per_match_red_1', 'wp_per_match_red_2', 'wp_per_match_blue_1', 'wp_per_match_blue_2'
        ]
        self.target_names = ['red_score', 'blue_score']
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        df = df.dropna()
        
        df_features = df[self.feature_names].copy()
        
        for metric in ['trueskill', 'opr', 'dpr', 'awp_per_match', 'ap_per_match', 'wp_per_match']:
            red_avg = (df_features[f'{metric}_red_1'] + df_features[f'{metric}_red_2']) / 2
            blue_avg = (df_features[f'{metric}_blue_1'] + df_features[f'{metric}_blue_2']) / 2
            df_features[f'{metric}_diff'] = red_avg - blue_avg
        
        for metric in ['trueskill', 'opr', 'dpr', 'awp_per_match', 'ap_per_match', 'wp_per_match']:
            df_features[f'{metric}_red_total'] = df_features[f'{metric}_red_1'] + df_features[f'{metric}_red_2']
            df_features[f'{metric}_blue_total'] = df_features[f'{metric}_blue_1'] + df_features[f'{metric}_blue_2']
        
        X = df_features.values
        y = df[self.target_names].values
        
        print(f"Data shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Original features: {len(self.feature_names)}")
        print(f"Total features after engineering: {X.shape[1]}")
        
        return X, y
    
    def train_scaler(self, X_train: np.ndarray, y_train: np.ndarray):
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train)
    
    def scale_features(self, X: np.ndarray) -> np.ndarray:
        return self.feature_scaler.transform(X)
    
    def scale_targets(self, y: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(y)
    
    def inverse_scale_targets(self, y: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(y)


class MatchScoreModelEvaluator:
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        results = {
            'overall_mae': mean_absolute_error(y_true, y_pred),
            'overall_mse': mean_squared_error(y_true, y_pred),
            'overall_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'overall_r2': r2_score(y_true, y_pred),
        }
        
        pred_wins = y_pred[:, 1] > y_pred[:, 0]  
        true_wins = y_true[:, 1] > y_true[:, 0]
        results['win_accuracy'] = accuracy_score(true_wins, pred_wins)
        
        results.update({
            'red_mae': mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
            'red_mse': mean_squared_error(y_true[:, 0], y_pred[:, 0]),
            'red_rmse': np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])),
            'red_r2': r2_score(y_true[:, 0], y_pred[:, 0]),
        })
        
        results.update({
            'blue_mae': mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
            'blue_mse': mean_squared_error(y_true[:, 1], y_pred[:, 1]),
            'blue_rmse': np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])),
            'blue_r2': r2_score(y_true[:, 1], y_pred[:, 1]),
        })
        
        return results

    @staticmethod
    def plot_training_history(history: Dict[str, list]):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training History - Match Score Prediction", fontsize=16)
        
        axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss (Combined)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history['train_mae'], label='Training MAE', color='blue')
        axes[0, 1].plot(history['val_mae'], label='Validation MAE', color='red')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(history['train_win_acc'], label='Training Win Acc', color='blue')
        axes[1, 0].plot(history['val_win_acc'], label='Validation Win Acc', color='red')
        axes[1, 0].set_title('Win Prediction Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history['train_mae'], label='Train MAE', color='blue', linestyle='-')
        axes[1, 1].plot(history['val_mae'], label='Val MAE', color='red', linestyle='-')
        axes[1, 1].set_title('MAE Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('_match_score_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
        """Plot actual vs predicted scores with win predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, color='red')
        axes[0, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                       [y_true[:, 0].min(), y_true[:, 0].max()], 'k--', lw=2)
        axes[0, 0].set_xlabel('Actual Red Score')
        axes[0, 0].set_ylabel('Predicted Red Score')
        axes[0, 0].set_title('Red Score Predictions')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6, color='blue')
        axes[0, 1].plot([y_true[:, 1].min(), y_true[:, 1].max()], 
                       [y_true[:, 1].min(), y_true[:, 1].max()], 'k--', lw=2)
        axes[0, 1].set_xlabel('Actual Blue Score')
        axes[0, 1].set_ylabel('Predicted Blue Score')
        axes[0, 1].set_title('Blue Score Predictions')
        axes[0, 1].grid(True, alpha=0.3)
        
        pred_wins = y_pred[:, 1] > y_pred[:, 0]
        true_wins = y_true[:, 1] > y_true[:, 0]
        
        colors = ['green' if pred == true else 'red' for pred, true in zip(pred_wins, true_wins)]
        
        axes[1, 0].scatter(y_true[:, 0] - y_true[:, 1], y_pred[:, 0] - y_pred[:, 1], 
                          alpha=0.6, c=colors)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Actual Score Difference (Red - Blue)')
        axes[1, 0].set_ylabel('Predicted Score Difference (Red - Blue)')
        axes[1, 0].set_title('Win Predictions (Green=Correct, Red=Wrong)')
        axes[1, 0].grid(True, alpha=0.3)
        
        score_diff_actual = y_true[:, 0] - y_true[:, 1]
        score_diff_pred = y_pred[:, 0] - y_pred[:, 1]
        
        axes[1, 1].hist(score_diff_actual, bins=30, alpha=0.7, label='Actual', color='blue')
        axes[1, 1].hist(score_diff_pred, bins=30, alpha=0.7, label='Predicted', color='red')
        axes[1, 1].set_xlabel('Score Difference (Red - Blue)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Score Difference Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('_match_score_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()


def main(df: pd.DataFrame) -> Dict[str, Any]:
    
    data_setup = MatchScoreDataSetup(use_robust_scaler=True)
    X, y = data_setup.prepare_features(df)
    
    wins = (y[:, 1] > y[:, 0]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4321, stratify=wins
    )
    
    train_wins = (y_train[:, 1] > y_train[:, 0]).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=4321, stratify=train_wins
    )
    
    data_setup.train_scaler(X_train, y_train)
    X_train_scaled = data_setup.scale_features(X_train)
    X_val_scaled = data_setup.scale_features(X_val)
    X_test_scaled = data_setup.scale_features(X_test)
    
    y_train_scaled = data_setup.scale_targets(y_train)
    y_val_scaled = data_setup.scale_targets(y_val)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Validation set shape: {X_val_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    print(f"\nWin distribution:")
    print(f"Train - Red wins: {np.mean(y_train[:, 0] > y_train[:, 1]):.3f}")
    print(f"Val   - Red wins: {np.mean(y_val[:, 0] > y_val[:, 1]):.3f}")
    print(f"Test  - Red wins: {np.mean(y_test[:, 0] > y_test[:, 1]):.3f}")
    
    rngs = nnx.Rngs(42)
    nn_model = MatchScoreNeuralNetwork(rngs, input_dim=X_train_scaled.shape[1])
    trainer = MatchScoreTrainer(nn_model, learning_rate=0.001)
    
    print("\nTraining  Match Score Neural Network...")
    nn_history = trainer.train(
        X_train_scaled, y_train_scaled, 
        X_val_scaled, y_val_scaled,
        epochs=150,
        batch_size=64,
        early_stop=15,
        min_delta=0.0005
    )
    
    y_pred_scaled = trainer.predict(X_test_scaled)
    y_pred = data_setup.inverse_scale_targets(y_pred_scaled)
    
    evaluator = MatchScoreModelEvaluator()
    results = evaluator.evaluate_model(y_test, y_pred)
    
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS -  MODEL")
    print("=" * 80)
    print(f"Overall Performance:")
    print(f"  MAE: {results['overall_mae']:.4f}")
    print(f"  MSE: {results['overall_mse']:.4f}")
    print(f"  RMSE: {results['overall_rmse']:.4f}")
    print(f"  R²: {results['overall_r2']:.4f}")
    print(f"  Win Accuracy: {results['win_accuracy']:.4f}")
    
    print(f"\nRed Score Performance:")
    print(f"  MAE: {results['red_mae']:.4f}")
    print(f"  MSE: {results['red_mse']:.4f}")
    print(f"  RMSE: {results['red_rmse']:.4f}")
    print(f"  R²: {results['red_r2']:.4f}")
    print(f"  MAPE: {results['red_mape']:.2f}%")
    
    print(f"\nBlue Score Performance:")
    print(f"  MAE: {results['blue_mae']:.4f}")
    print(f"  MSE: {results['blue_mse']:.4f}")
    print(f"  RMSE: {results['blue_rmse']:.4f}")
    print(f"  R²: {results['blue_r2']:.4f}")
    print(f"  MAPE: {results['blue_mape']:.2f}%")
    
    evaluator.plot_training_history(nn_history)
    evaluator.plot_predictions(y_test, y_pred)
    
    print("\nSample Predictions with Win Outcomes:")
    print("-" * 60)
    for i in range(min(15, len(y_test))):
        true_winner = "Blue" if y_test[i][1] > y_test[i][0] else "Red"
        pred_winner = "Blue" if y_pred[i][1] > y_pred[i][0] else "Red"
        correct = ":D" if true_winner == pred_winner else ":("
        
        print(f"Match {i+1}: {correct}")
        print(f"  Actual:    Red={y_test[i][0]:.1f}, Blue={y_test[i][1]:.1f} → {true_winner} wins")
        print(f"  Predicted: Red={y_pred[i][0]:.1f}, Blue={y_pred[i][1]:.1f} → {pred_winner} wins")
        print()
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    true_diff = y_test[:, 0] - y_test[:, 1]
    pred_diff = y_pred[:, 0] - y_pred[:, 1]
    
    print(f"Score Difference Analysis:")
    print(f"  True difference mean: {np.mean(true_diff):.2f} ± {np.std(true_diff):.2f}")
    print(f"  Pred difference mean: {np.mean(pred_diff):.2f} ± {np.std(pred_diff):.2f}")
    print(f"  Difference correlation: {np.corrcoef(true_diff, pred_diff)[0,1]:.4f}")

    
    return {
        'model': trainer,
        'results': results,
        'history': nn_history,
        'data_setup': data_setup,
        'test_data': (X_test_scaled, y_test, y_pred)
    }


data = pd.read_csv('swapped.csv')
results = main(data)

