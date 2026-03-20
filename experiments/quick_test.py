import pickle, numpy as np, sys, time
from advanced_predictor import augment_features, TARGETS
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

def p(*a): print(*a, flush=True)

p("Loading data...")
with open('cache_train_features.pkl','rb') as f: d = pickle.load(f)
X_tr, Y_tr, df_tr = d['X'], d['Y'], d['df']
with open('cache_test_features.pkl','rb') as f: d = pickle.load(f)
X_te, Y_te = d['X'], d['Y']
p(f"Train {X_tr.shape}  Test {X_te.shape}")

X_aug = augment_features(X_tr)
X_te_aug = augment_features(X_te)
sc = StandardScaler()
X_sc = sc.fit_transform(X_aug)
X_te_sc = sc.transform(X_te_aug)
p(f"Augmented: {X_aug.shape[1]} features")

for tname in TARGETS:
    j = TARGETS.index(tname)
    t0 = time.time()
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=63,
                           min_child_samples=20, subsample=0.8,
                           colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    m.fit(X_sc, Y_tr[:,j])
    yp = m.predict(X_te_sc)
    mae = mean_absolute_error(Y_te[:,j], yp)
    r2 = r2_score(Y_te[:,j], yp)
    rho,_ = spearmanr(Y_te[:,j], yp)
    ok = 'PASS<0.10' if mae < 0.10 else f'MAE={mae:.4f}'
    p(f"  {ok}  {tname}  R2={r2:.4f}  rho={rho:.3f}  ({time.time()-t0:.0f}s)")
    p(f"    true: {Y_te[:,j].round(3).tolist()}")
    p(f"    pred: {yp.round(3).tolist()}")

p("DONE")
