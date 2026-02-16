import os
import time
import pickle
import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

CACHE_DIR = Path("cache_radical")
RESULTS_DIR = Path("results_radical")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (48, 48)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("FULL DATASET - RADICAL APPROACH")
print("Data Augmentation + Heavy Regularization + Diverse Ensemble")
print("Target: 70-73% based on 3K test")
print("="*70)


def augment_image(img):
    augmented = [img]
    augmented.append(cv2.flip(img, 1))
    
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (48, 48))
        augmented.append(rotated)
    
    for delta in [-20, 20]:
        adjusted = np.clip(img.astype(int) + delta, 0, 255).astype(np.uint8)
        augmented.append(adjusted)
    
    return augmented


def radical_preprocess(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    img = clahe.apply(img)
    img = cv2.fastNlMeansDenoising(img, h=15)
    edges = cv2.Canny(img, 50, 150)
    img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def extract_deep_features(img):
    features = []
    
    for patch_size in [6, 12, 24]:
        step = patch_size
        for i in range(0, 48-patch_size+1, step):
            for j in range(0, 48-patch_size+1, step):
                patch = img[i:i+patch_size, j:j+patch_size]
                features.extend([
                    float(np.mean(patch)),
                    float(np.std(patch)),
                    float(np.max(patch) - np.min(patch))
                ])
    
    eyes = img[10:26, :]
    mouth = img[30:46, 10:38]
    
    for region in [eyes, mouth]:
        if region.size > 0:
            features.extend([
                float(np.mean(region)),
                float(np.std(region)),
                float(np.median(region)),
                float(np.percentile(region, 25)),
                float(np.percentile(region, 75)),
                float(np.min(region)),
                float(np.max(region))
            ])
    
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.hypot(sobel_x, sobel_y)
    
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_mag = np.hypot(scharr_x, scharr_y)
    
    for grad in [sobel_x, sobel_y, sobel_mag, scharr_x, scharr_y, scharr_mag]:
        features.extend([
            float(np.mean(grad)),
            float(np.std(grad)),
            float(np.percentile(np.abs(grad), 90))
        ])
    
    for bins in [16, 32]:
        hist = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        features.extend(hist.astype(float).tolist())
    
    hog = cv2.HOGDescriptor(_winSize=(48,48), _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)
    hog_feat = hog.compute(img)
    if hog_feat is not None:
        features.extend(hog_feat.flatten().astype(float).tolist())
    
    lbp = []
    for i in range(1, 47):
        for j in range(1, 47):
            center = img[i, j]
            code = 0
            code |= (img[i-1,j-1] >= center) << 0
            code |= (img[i-1,j] >= center) << 1
            code |= (img[i-1,j+1] >= center) << 2
            code |= (img[i,j+1] >= center) << 3
            code |= (img[i+1,j+1] >= center) << 4
            code |= (img[i+1,j] >= center) << 5
            code |= (img[i+1,j-1] >= center) << 6
            code |= (img[i,j-1] >= center) << 7
            lbp.append(code)
    
    lbp_hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
    lbp_hist = lbp_hist.astype(float) / (len(lbp) + 1e-8)
    features.extend(lbp_hist.tolist())
    
    return np.array(features, dtype=np.float32)


def load_and_process(train_dir='train', test_dir='test'):
    cache_file = CACHE_DIR / "processed_data.npz"
    
    if cache_file.exists():
        print("\nLoading cached processed data")
        data = np.load(cache_file)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test'], data['emotions']
    
    print("\nLoading dataset")
    
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    
    emotions = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Emotions: {emotions}")
    
    for emotion in emotions:
        train_path = os.path.join(train_dir, emotion)
        count = 0
        for img_file in os.listdir(train_path):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(train_path, img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    train_imgs.append(img)
                    train_labels.append(emotion.lower().strip())
                    count += 1
        print(f"  {emotion}: {count} train")
        
        test_path = os.path.join(test_dir, emotion)
        if os.path.exists(test_path):
            count = 0
            for img_file in os.listdir(test_path):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img = cv2.imread(os.path.join(test_path, img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        test_imgs.append(img)
                        test_labels.append(emotion.lower().strip())
                        count += 1
            print(f"  {emotion}: {count} test")
    
    print(f"\nTotal: {len(train_imgs)} train, {len(test_imgs)} test")
    
    print("\nPreprocessing + Augmentation (4x)")
    train_processed = []
    train_labels_aug = []
    
    start = time.time()
    for idx, (img, label) in enumerate(zip(train_imgs, train_labels)):
        preprocessed = radical_preprocess(img)
        
        train_processed.append(preprocessed)
        train_labels_aug.append(label)
        
        augmented = augment_image(preprocessed)
        for aug in augmented[1:4]:
            train_processed.append(aug)
            train_labels_aug.append(label)
        
        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            remaining = (len(train_imgs) - idx - 1) / rate
            print(f"  {idx+1}/{len(train_imgs)} → {len(train_processed)} total - ETA: {remaining/60:.1f} min")
    
    print(f"  Final: {len(train_processed)} train samples (4x augmented)")
    
    print("\nPreprocessing test set")
    test_processed = []
    for idx, img in enumerate(test_imgs):
        preprocessed = radical_preprocess(img)
        test_processed.append(preprocessed)
        
        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1}/{len(test_imgs)}")
    
    print("\nExtracting features")
    train_features = []
    start = time.time()
    
    for idx, img in enumerate(train_processed):
        feat = extract_deep_features(img)
        train_features.append(feat)
        
        if (idx + 1) % 10000 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            remaining = (len(train_processed) - idx - 1) / rate
            print(f"  Train: {idx+1}/{len(train_processed)} ({(idx+1)/len(train_processed)*100:.0f}%) - "
                  f"{rate:.0f} img/s - ETA: {remaining/60:.1f} min")
    
    X_train = np.vstack(train_features)
    
    test_features = []
    for idx, img in enumerate(test_processed):
        feat = extract_deep_features(img)
        test_features.append(feat)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Test: {idx+1}/{len(test_processed)}")
    
    X_test = np.vstack(test_features)
    
    print(f"\nFeatures: Train {X_train.shape}, Test {X_test.shape}")
    
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels_aug)
    y_test = le.transform(test_labels)
    
    np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, 
                       X_test=X_test, y_test=y_test, emotions=emotions)
    print("Cached processed data")
    
    return X_train, y_train, X_test, y_test, emotions


X_train, y_train, X_test, y_test, emotions = load_and_process()

emotions_clean = [e.lower().strip() for e in emotions]

print("\nFeature engineering")
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e9, neginf=-1e9)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e9, neginf=-1e9)

cache_selector = CACHE_DIR / "selector.pkl"
cache_scaler = CACHE_DIR / "scaler.pkl"

if cache_selector.exists():
    with open(cache_selector, 'rb') as f:
        selector = pickle.load(f)
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
else:
    print("  Feature selection")
    selector = SelectKBest(f_classif, k=min(400, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    with open(cache_selector, 'wb') as f:
        pickle.dump(selector, f)

if cache_scaler.exists():
    with open(cache_scaler, 'rb') as f:
        scaler = pickle.load(f)
    X_train_scaled = scaler.transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
else:
    print("  Scaling")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    with open(cache_scaler, 'wb') as f:
        pickle.dump(scaler, f)

print(f"  Final: {X_train_scaled.shape[1]} features")

print("\n" + "="*70)
print("TRAINING DIVERSE ENSEMBLE")
print("="*70)

models_data = {}

# Random Forest
cache_rf = RESULTS_DIR / "rf.pkl"
if cache_rf.exists():
    print("\n1/4: Loading Random Forest")
    with open(cache_rf, 'rb') as f:
        models_data['RF'] = pickle.load(f)
else:
    print("\n1/4: Training Random Forest")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    print("  5-fold CV")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
        rf.fit(X_train_scaled[train_idx], y_train[train_idx])
        score = rf.score(X_train_scaled[val_idx], y_train[val_idx])
        cv_scores.append(score)
        print(f"    Fold {fold}: {score*100:.2f}%")
    
    print(f"  CV: {np.mean(cv_scores)*100:.2f}%")
    
    rf.fit(X_train_scaled, y_train)
    models_data['RF'] = {'model': rf, 'cv': np.mean(cv_scores)}
    
    with open(cache_rf, 'wb') as f:
        pickle.dump(models_data['RF'], f)

# Gradient Boosting
cache_gb = RESULTS_DIR / "gb.pkl"
if cache_gb.exists():
    print("\n2/4: Loading Gradient Boosting")
    with open(cache_gb, 'rb') as f:
        models_data['GB'] = pickle.load(f)
else:
    print("\n2/4: Training Gradient Boosting")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    gb.fit(X_train_scaled, y_train)
    models_data['GB'] = {'model': gb}
    
    with open(cache_gb, 'wb') as f:
        pickle.dump(models_data['GB'], f)

# SVM
cache_svm = RESULTS_DIR / "svm.pkl"
if cache_svm.exists():
    print("\n3/4: Loading SVM")
    with open(cache_svm, 'rb') as f:
        models_data['SVM'] = pickle.load(f)
else:
    print("\n3/4: Training SVM (this may take 10-20 min)")
    svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    models_data['SVM'] = {'model': svm}
    
    with open(cache_svm, 'wb') as f:
        pickle.dump(models_data['SVM'], f)

# XGBoost
if HAS_XGB:
    cache_xgb = RESULTS_DIR / "xgb.pkl"
    if cache_xgb.exists():
        print("\n4/4: Loading XGBoost")
        with open(cache_xgb, 'rb') as f:
            models_data['XGB'] = pickle.load(f)
    else:
        print("\n4/4: Training XGBoost")
        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        xgb_model.fit(X_train_scaled, y_train)
        models_data['XGB'] = {'model': xgb_model}
        
        with open(cache_xgb, 'wb') as f:
            pickle.dump(models_data['XGB'], f)

print("\n" + "="*70)
print("EVALUATION")
print("="*70)

results = {}

for name, data in models_data.items():
    model = data['model']
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{name}: {acc*100:.2f}%")
    
    report = classification_report(y_test, y_pred, target_names=emotions_clean, output_dict=True)
    
    for emotion in emotions_clean:
        p = report[emotion]['precision']
        r = report[emotion]['recall']
        print(f"  {emotion:<10} P:{p:.3f} R:{r:.3f}")
    
    results[name] = acc

print("\n" + "="*70)
print("ENSEMBLE")
print("="*70)

y_proba_sum = np.zeros((len(X_test_scaled), len(emotions_clean)))

for name, data in models_data.items():
    y_proba_sum += data['model'].predict_proba(X_test_scaled)

y_pred_ensemble = np.argmax(y_proba_sum, axis=1)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)

print(f"\nEnsemble: {ensemble_acc*100:.2f}%")

report_ens = classification_report(y_test, y_pred_ensemble, target_names=emotions_clean, output_dict=True)

for emotion in emotions_clean:
    p = report_ens[emotion]['precision']
    r = report_ens[emotion]['recall']
    print(f"  {emotion:<10} P:{p:.3f} R:{r:.3f}")

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"\n{'Model':<15} {'Test Accuracy':<15}")
print("-"*70)
for name, acc in results.items():
    print(f"{name:<15} {acc*100:<14.2f}%")
print(f"{'Ensemble':<15} {ensemble_acc*100:<14.2f}%")

best = max(results.items(), key=lambda x: x[1])
print(f"\nBest single: {best[0]} - {best[1]*100:.2f}%")
print(f"Ensemble: {ensemble_acc*100:.2f}%")

print("\n" + "="*70)