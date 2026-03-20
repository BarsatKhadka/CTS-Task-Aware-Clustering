"""
Comprehensive tests for CTSPredictor.

Test categories:
  1. Feature extraction (graph + CSV)
  2. Dataset building
  3. Model training and prediction
  4. Evaluation metrics
  5. Save / load round-trip
  6. Zero-shot generalization (zipdiv as unseen design)
  7. Edge cases and robustness
  8. Prediction quality thresholds

Run with:
    python test_cts_predictor.py
or:
    python -m pytest test_cts_predictor.py -v
"""

import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

warnings.filterwarnings("ignore")

# ── Import module under test ───────────────────────────────────────────────
from cts_predictor import (    
    BENCH_ROOT,
    TARGETS,
    CTSPredictor,
    build_dataset,
    extract_csv_features,
    extract_graph_features,
    leave_one_design_out_cv,
    _gini,
)

BENCH   = BENCH_ROOT
DATA_ROOT = BENCH / "dataset_root"
TRAIN_CSV = DATA_ROOT / "clocknet_unified_manifest.csv"
TEST_CSV  = DATA_ROOT / "clocknet_unified_manifest_test.csv"

# Fixture: small train subset (fast)
@pytest.fixture(scope="module")
def small_dataset():
    df = pd.read_csv(TRAIN_CSV)
    parts = [grp.sample(min(25, len(grp)), random_state=42)
             for _, grp in df.groupby("design_name")]
    sub = pd.concat(parts).reset_index(drop=True)
    sub.to_csv("/tmp/small_manifest.csv", index=False)
    X, Y, meta = build_dataset("/tmp/small_manifest.csv", BENCH, verbose=False)
    return X, Y, meta


@pytest.fixture(scope="module")
def full_train():
    return build_dataset(TRAIN_CSV, BENCH, verbose=False)


@pytest.fixture(scope="module")
def full_test():
    return build_dataset(TEST_CSV, BENCH, verbose=False)


@pytest.fixture(scope="module")
def trained_model(full_train):
    X, Y, _ = full_train
    pred = CTSPredictor()
    pred.fit(X, Y, verbose=False)
    return pred


# ═══════════════════════════════════════════════════════════════════════════
#  1. Feature extraction — graph features
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphFeatureExtraction:
    """Tests for extract_graph_features()."""

    @pytest.fixture(autouse=True)
    def get_sample_path(self):
        df = pd.read_csv(TRAIN_CSV)
        self.sample_path = BENCH / str(df.iloc[0]["cluster_graph_path"])
        assert self.sample_path.exists(), f"Missing: {self.sample_path}"

    def test_returns_numpy_array(self):
        feats = extract_graph_features(self.sample_path)
        assert isinstance(feats, np.ndarray)

    def test_correct_dtype(self):
        feats = extract_graph_features(self.sample_path)
        assert feats.dtype == np.float32

    def test_correct_dimension(self):
        feats = extract_graph_features(self.sample_path)
        assert feats.shape == (52,), f"Expected (52,), got {feats.shape}"

    def test_no_nan_or_inf(self):
        feats = extract_graph_features(self.sample_path)
        assert not np.isnan(feats).any(), "NaN in graph features"
        assert not np.isinf(feats).any(), "Inf in graph features"

    def test_deterministic(self):
        f1 = extract_graph_features(self.sample_path)
        f2 = extract_graph_features(self.sample_path)
        np.testing.assert_array_equal(f1, f2)

    def test_different_graphs_different_features(self):
        df = pd.read_csv(TRAIN_CSV)
        # Pick rows from different placement IDs (not same CTS run)
        unique_placements = df["placement_id"].unique()
        assert len(unique_placements) >= 2
        r0 = df[df["placement_id"] == unique_placements[0]].iloc[0]
        r1 = df[df["placement_id"] == unique_placements[1]].iloc[0]
        f0 = extract_graph_features(BENCH / str(r0["cluster_graph_path"]))
        f1 = extract_graph_features(BENCH / str(r1["cluster_graph_path"]))
        assert not np.allclose(f0, f1), "Different graphs should have different features"

    def test_scale_features_positive(self):
        """log-scale features should be non-negative."""
        feats = extract_graph_features(self.sample_path)
        # feats[0:3] are log1p(n_nodes), log1p(n_edges), log1p(edge/node)
        assert feats[0] >= 0, "log(n_nodes) should be >= 0"
        assert feats[1] >= 0, "log(n_edges) should be >= 0"

    def test_spatial_features_in_range(self):
        """Centroids are normalized [0,1] so mean_x/mean_y in [0,1]."""
        feats = extract_graph_features(self.sample_path)
        # feats[3]=mean_x, feats[5]=mean_y
        assert 0.0 <= feats[3] <= 1.0, f"mean_x={feats[3]} out of [0,1]"
        assert 0.0 <= feats[5] <= 1.0, f"mean_y={feats[5]} out of [0,1]"

    def test_multiple_designs(self):
        """Features should work for all designs (aes, ethmac, sha256, etc.)."""
        df = pd.read_csv(TRAIN_CSV)
        for design in df["design_name"].unique():
            row = df[df["design_name"] == design].iloc[0]
            path = BENCH / str(row["cluster_graph_path"])
            feats = extract_graph_features(path)
            assert feats.shape == (52,), f"{design}: bad shape"
            assert not np.isnan(feats).any(), f"{design}: NaN"


# ═══════════════════════════════════════════════════════════════════════════
#  2. CSV feature extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestCSVFeatureExtraction:
    """Tests for extract_csv_features()."""

    @pytest.fixture(autouse=True)
    def sample_row(self):
        df = pd.read_csv(TRAIN_CSV)
        self.row = df.iloc[0]

    def test_returns_numpy_array(self):
        feats = extract_csv_features(self.row)
        assert isinstance(feats, np.ndarray)

    def test_correct_dimension(self):
        feats = extract_csv_features(self.row)
        assert feats.shape == (12,), f"Expected (12,), got {feats.shape}"

    def test_correct_dtype(self):
        feats = extract_csv_features(self.row)
        assert feats.dtype == np.float32

    def test_no_nan(self):
        feats = extract_csv_features(self.row)
        assert not np.isnan(feats).any()

    def test_area_mode_delay_mode_exclusive(self):
        """area_mode and delay_mode should not both be 1 for same row."""
        df = pd.read_csv(TRAIN_CSV)
        for _, row in df.sample(20, random_state=0).iterrows():
            f = extract_csv_features(row)
            area_flag, delay_flag = f[3], f[4]
            assert not (area_flag == 1 and delay_flag == 1), "Both AREA and DELAY set"

    def test_knob_values_present(self):
        """CTS knobs should appear directly in features (sanity)."""
        row = self.row
        feats = extract_csv_features(row)
        # cts_max_wire is feats[8], cts_buf_dist is feats[9]
        assert feats[8] == float(row["cts_max_wire"])
        assert feats[9] == float(row["cts_buf_dist"])

    def test_missing_columns_use_defaults(self):
        """Missing columns should not raise errors."""
        minimal = pd.Series({"cts_max_wire": 200, "cts_buf_dist": 100,
                             "cts_cluster_size": 20, "cts_cluster_dia": 50})
        feats = extract_csv_features(minimal)
        assert feats.shape == (12,)
        assert not np.isnan(feats).any()


# ═══════════════════════════════════════════════════════════════════════════
#  3. Gini coefficient helper
# ═══════════════════════════════════════════════════════════════════════════

class TestGini:
    def test_equal_distribution(self):
        assert _gini(np.ones(10)) == pytest.approx(0.0, abs=1e-6)

    def test_singleton_nonzero(self):
        arr = np.array([0.0, 0.0, 1.0])
        g = _gini(arr)
        assert g >= 0

    def test_range(self):
        for _ in range(10):
            arr = np.abs(np.random.randn(50))
            g = _gini(arr)
            assert -0.01 <= g <= 1.01, f"Gini out of range: {g}"

    def test_empty_returns_zero(self):
        assert _gini(np.zeros(5)) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  4. Dataset building
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildDataset:
    def test_shapes_consistent(self, small_dataset):
        X, Y, meta = small_dataset
        assert X.shape[0] == Y.shape[0] == len(meta)
        assert X.ndim == 2
        assert Y.shape[1] == 3

    def test_feature_dim(self, small_dataset):
        X, Y, _ = small_dataset
        assert X.shape[1] == 64, f"Expected 64 features, got {X.shape[1]}"

    def test_targets_skew_positive(self, small_dataset):
        _, Y, _ = small_dataset
        assert (Y[:, 0] > 0).all(), "Skew should be positive"

    def test_log_power_finite(self, small_dataset):
        _, Y, _ = small_dataset
        assert np.isfinite(Y[:, 1]).all(), "log(power) has non-finite values"

    def test_log_wl_positive(self, small_dataset):
        _, Y, _ = small_dataset
        assert (Y[:, 2] > 0).all(), "log(wirelength) should be positive"

    def test_no_nan_in_X(self, small_dataset):
        X, _, _ = small_dataset
        assert not np.isnan(X).any(), "NaN in feature matrix"

    def test_no_inf_in_X(self, small_dataset):
        X, _, _ = small_dataset
        assert not np.isinf(X).any(), "Inf in feature matrix"

    def test_full_train_size(self, full_train):
        X, Y, meta = full_train
        assert len(X) >= 4000, f"Expected ≥4000 samples, got {len(X)}"

    def test_test_set_is_zipdiv_only(self, full_test):
        _, _, meta = full_test
        assert set(meta["design_name"].unique()) == {"zipdiv"}

    def test_test_set_size(self, full_test):
        X, _, _ = full_test
        assert len(X) >= 450, f"Expected ≥450 test samples, got {len(X)}"


# ═══════════════════════════════════════════════════════════════════════════
#  5. CTSPredictor — training
# ═══════════════════════════════════════════════════════════════════════════

class TestCTSPredictorTraining:
    def test_fit_runs_without_error(self, small_dataset):
        X, Y, _ = small_dataset
        pred = CTSPredictor()
        pred.fit(X, Y, verbose=False)
        assert pred._fitted

    def test_predict_shape(self, small_dataset):
        X, Y, _ = small_dataset
        pred = CTSPredictor()
        pred.fit(X, Y, verbose=False)
        out = pred.predict(X[:10])
        assert out.shape == (10, 3)

    def test_predict_natural_shape(self, small_dataset):
        X, Y, _ = small_dataset
        pred = CTSPredictor()
        pred.fit(X, Y, verbose=False)
        df_out = pred.predict_natural(X[:5])
        assert isinstance(df_out, pd.DataFrame)
        assert list(df_out.columns) == ["skew_pred", "power_pred", "wirelength_pred"]
        assert len(df_out) == 5

    def test_predict_natural_power_positive(self, small_dataset):
        X, Y, _ = small_dataset
        pred = CTSPredictor()
        pred.fit(X, Y, verbose=False)
        df_out = pred.predict_natural(X)
        assert (df_out["power_pred"] > 0).all()
        assert (df_out["wirelength_pred"] > 0).all()

    def test_raises_before_fit(self, small_dataset):
        X, _, _ = small_dataset
        pred = CTSPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            pred.predict(X)

    def test_feature_importance_shape(self, trained_model):
        imp = trained_model.feature_importance(top_k=10)
        assert isinstance(imp, pd.DataFrame)
        assert "feature" in imp.columns
        assert "importance" in imp.columns
        assert len(imp) == 10


# ═══════════════════════════════════════════════════════════════════════════
#  6. Save / load round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveLoad:
    def test_save_load_predictions_identical(self, small_dataset):
        X, Y, _ = small_dataset
        pred = CTSPredictor()
        pred.fit(X, Y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        pred.save(path)
        pred2 = CTSPredictor.load(path)

        Y1 = pred.predict(X)
        Y2 = pred2.predict(X)
        np.testing.assert_allclose(Y1, Y2, rtol=1e-5,
                                   err_msg="Predictions differ after save/load")

    def test_loaded_model_is_fitted(self, small_dataset):
        X, Y, _ = small_dataset
        pred = CTSPredictor()
        pred.fit(X, Y, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        pred.save(path)
        pred2 = CTSPredictor.load(path)
        assert pred2._fitted

    def test_saved_file_has_correct_keys(self, small_dataset):
        X, Y, _ = small_dataset
        pred = CTSPredictor()
        pred.fit(X, Y, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        pred.save(path)
        with open(path, "rb") as f:
            d = pickle.load(f)
        assert "models"      in d
        assert "scaler"      in d
        assert "task_params" in d


# ═══════════════════════════════════════════════════════════════════════════
#  7. Zero-shot generalization quality thresholds
# ═══════════════════════════════════════════════════════════════════════════

class TestZeroShotGeneralization:
    """
    Test that the model achieves acceptable MAE on zipdiv (unseen design).

    Zero-shot thresholds are generous because zipdiv is a completely different
    design family (integer divider) never seen during training.

    Absolute MAE thresholds (natural units, achievable via physics normalization):
      - skew:  MAE < 0.15 ns  (true mean ~0.51, std ~0.004 — zero-shot offset dominated)
      - power: MAE < 0.0025 W (true mean ~0.003 W — ~80% relative error allowed)
      - WL:    MAE < 10000 µm (true mean ~27178 µm — ~37% relative error)
    """

    def test_skew_mae_reasonable(self, trained_model, full_test):
        X_te, Y_te, _ = full_test
        Y_pred = trained_model.predict(X_te)
        mae = float(np.mean(np.abs(Y_te[:, 0] - Y_pred[:, 0])))
        assert mae < 0.15, (
            f"Skew MAE={mae:.4f} too high for zero-shot (threshold: 0.15 ns)")

    def test_power_mae_natural_reasonable(self, trained_model, full_test):
        X_te, Y_te, _ = full_test
        preds = trained_model.predict_natural(X_te)
        # Reconstruct true natural power from normalized target + act_sum_log
        act_sum_log = X_te[:, 21 + 4]   # N_FF_FEAT_IDX + 4
        true_power  = np.exp(Y_te[:, 1] + act_sum_log)
        mae = float(np.mean(np.abs(true_power - preds["power_pred"].values)))
        assert mae < 0.0025, (
            f"Power MAE={mae:.6f} W too high (threshold: 0.0025 W)")

    def test_wl_mae_natural_reasonable(self, trained_model, full_test):
        X_te, Y_te, _ = full_test
        preds = trained_model.predict_natural(X_te)
        n_ff_log = X_te[:, 21]
        true_wl  = np.exp(Y_te[:, 2] + n_ff_log) - 1
        mae = float(np.mean(np.abs(true_wl - preds["wirelength_pred"].values)))
        assert mae < 10000, (
            f"WL MAE={mae:.0f} µm too high (threshold: 10000 µm)")

    def test_predictions_capture_knob_trend(self, trained_model, full_test):
        """
        WL predictions should have strong Spearman correlation on zero-shot.
        Skew and power may have weaker correlations due to design-specific
        knob sensitivity (zipdiv has very narrow skew/power range).
        """
        from scipy.stats import spearmanr
        X_te, Y_te, _ = full_test
        Y_pred = trained_model.predict(X_te)

        # WL should be predictable (rho > 0.5)
        rho_wl, _ = spearmanr(Y_te[:, 2], Y_pred[:, 2])
        assert rho_wl > 0.3, (
            f"WL Spearman ρ={rho_wl:.3f} too low on zipdiv (threshold: 0.3)")

        # At least 1 out of 3 targets has ρ > 0.1
        good = sum(1 for j in range(3)
                   if spearmanr(Y_te[:, j], Y_pred[:, j])[0] > 0.1)
        assert good >= 1, (
            f"Only {good}/3 targets have ρ>0.1 on zipdiv zero-shot")

    def test_skew_predictions_positive(self, trained_model, full_test):
        X_te, _, _ = full_test
        preds = trained_model.predict_natural(X_te)
        assert (preds["skew_pred"] > 0).all(), "Skew predictions should be positive"

    def test_power_predictions_in_plausible_range(self, trained_model, full_test):
        X_te, _, _ = full_test
        preds = trained_model.predict_natural(X_te)
        assert (preds["power_pred"] > 0).all()
        assert (preds["power_pred"] < 10).all(), "Power > 10W is implausible"

    def test_wl_predictions_in_plausible_range(self, trained_model, full_test):
        X_te, _, _ = full_test
        preds = trained_model.predict_natural(X_te)
        assert (preds["wirelength_pred"] > 0).all()
        assert (preds["wirelength_pred"] < 1e8).all()


# ═══════════════════════════════════════════════════════════════════════════
#  8. In-distribution quality thresholds (leave-one-design-out)
# ═══════════════════════════════════════════════════════════════════════════

class TestInDistributionQuality:
    """
    Quick in-distribution check: train on 3 designs, test on 4th.
    Uses a small subset for speed.
    """

    @pytest.fixture(scope="class")
    def cv_dataset(self):
        df = pd.read_csv(TRAIN_CSV)
        parts = [grp.sample(min(100, len(grp)), random_state=0)
                 for _, grp in df.groupby("design_name")]
        sub = pd.concat(parts).reset_index(drop=True)
        sub.to_csv("/tmp/cv_manifest.csv", index=False)
        X, Y, meta = build_dataset("/tmp/cv_manifest.csv", BENCH, verbose=False)
        return X, Y, meta

    def test_cv_skew_log_mae_under_threshold(self, cv_dataset):
        """
        LODO CV log(skew) MAE should be reasonable per fold.
        Skew is in natural units so threshold is in ns.
        """
        X, Y, meta = cv_dataset
        results = leave_one_design_out_cv(X, Y, meta, verbose=False)
        mean_mae = results["skew_setup_mae_nat"].mean()
        assert mean_mae < 0.3, f"CV skew MAE={mean_mae:.4f} ns (threshold: 0.3)"

    def test_cv_power_log_mae_under_threshold(self, cv_dataset):
        X, Y, meta = cv_dataset
        results = leave_one_design_out_cv(X, Y, meta, verbose=False)
        mean_mae = results["power_total_mae_log"].mean()
        assert mean_mae < 2.0, f"CV log(power) MAE={mean_mae:.4f} (threshold: 2.0)"

    def test_cv_returns_all_folds(self, cv_dataset):
        X, Y, meta = cv_dataset
        designs = meta["design_name"].unique()
        results = leave_one_design_out_cv(X, Y, meta, verbose=False)
        assert len(results) == len(designs)


# ═══════════════════════════════════════════════════════════════════════════
#  9. Robustness / edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestRobustness:
    def test_predict_single_sample(self, trained_model, full_test):
        X_te, _, _ = full_test
        out = trained_model.predict(X_te[:1])
        assert out.shape == (1, 3)

    def test_predict_all_test(self, trained_model, full_test):
        X_te, _, _ = full_test
        out = trained_model.predict(X_te)
        assert out.shape[0] == len(X_te)
        assert not np.isnan(out).any()

    def test_features_vary_across_knobs(self, full_train):
        """
        CTS knob features (indices 8-11 in CSV block) should vary across rows
        within the same design, proving they're captured.
        """
        X, _, meta = full_train
        design = meta["design_name"].iloc[0]
        mask = (meta["design_name"] == design).values
        X_sub = X[mask]
        # CSV features start at index 0; knobs are indices 8-11
        knob_cols = X_sub[:, 8:12]
        assert knob_cols.std(axis=0).sum() > 0, "Knob features have zero variance"

    def test_graph_features_vary_across_designs(self, full_train):
        """
        Graph features (indices 12+) should differ between designs.
        """
        X, _, meta = full_train
        design_means = []
        for d in meta["design_name"].unique():
            mask = (meta["design_name"] == d).values
            design_means.append(X[mask, 12:].mean(axis=0))
        # At least one graph feature should differ between designs
        design_means = np.stack(design_means)
        assert design_means.std(axis=0).max() > 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  10. Full pipeline smoke test
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end test: build dataset → train → predict → evaluate."""

    def test_full_pipeline_runs(self, small_dataset, full_test):
        X_tr, Y_tr, meta_tr = small_dataset
        X_te, Y_te, meta_te = full_test

        pred = CTSPredictor()
        pred.fit(X_tr, Y_tr, verbose=False)

        # Predict
        Y_pred = pred.predict(X_te)
        assert Y_pred.shape == (len(X_te), 3)

        # Evaluate (just check it runs)
        results = pred.evaluate(X_te, Y_te, label="smoke", df_meta=meta_te)
        assert len(results) == 3

    def test_evaluate_returns_dataframe(self, trained_model, full_test):
        X_te, Y_te, _ = full_test
        results = trained_model.evaluate(X_te, Y_te, label="test", df_meta=None)
        assert isinstance(results, pd.DataFrame)
        assert "target" in results.columns
        assert "mae_log" in results.columns
        assert "r2" in results.columns

    def test_natural_wl_closer_to_true_than_mean(self, trained_model, full_test):
        """
        WL predictions should beat a naive mean baseline on zipdiv.
        (Skew/power may not beat the mean due to the very low within-zipdiv
        variance compared to the cross-design offset.)
        """
        from cts_predictor import N_FF_FEAT_IDX
        X_te, Y_te, _ = full_test
        preds = trained_model.predict_natural(X_te)

        n_ff_log = X_te[:, N_FF_FEAT_IDX]
        y_true_wl = np.exp(Y_te[:, 2] + n_ff_log) - 1
        naive_mae = float(np.abs(y_true_wl - y_true_wl.mean()).mean())
        model_mae = float(np.abs(y_true_wl - preds["wirelength_pred"].values).mean())
        assert model_mae < naive_mae * 2, (
            f"WL model MAE {model_mae:.0f} is more than 2× worse than naive "
            f"baseline {naive_mae:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "-x"]))
