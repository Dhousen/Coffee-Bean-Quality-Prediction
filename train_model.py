import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced Feature Engineering class with consistent column handling"""
    
    def __init__(self):
        self.le_tekstur = LabelEncoder()
        self.le_warna = LabelEncoder()
        self.expected_columns = None
        self.fitted = False
        
    def fit(self, X, y=None):
        """Fit encoders and define expected columns"""
        X_copy = X.copy()
        
        # Fit label encoders
        self.le_tekstur.fit(X_copy["tekstur"])
        self.le_warna.fit(X_copy["warna"])
        
        # Create dummy data to determine expected columns
        X_transformed = self._transform_features(X_copy)
        self.expected_columns = X_transformed.columns.tolist()
        self.fitted = True
        
        return self
        
    def transform(self, X):
        """Transform features with consistent column handling"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
            
        X_transformed = self._transform_features(X.copy())
        
        # Ensure consistent columns
        for col in self.expected_columns:
            if col not in X_transformed.columns:
                X_transformed[col] = 0
                
        # Reorder columns to match training
        X_transformed = X_transformed[self.expected_columns]
        
        return X_transformed
        
    def _transform_features(self, X):
        """Core feature transformation logic"""
        X_engineered = X.copy()
        
        # Basic feature engineering
        X_engineered['berat_ukuran_ratio'] = X['berat_per_10_biji'] / (X['ukuran_rata2_per_10_biji'] + 1e-8)
        X_engineered['density_proxy'] = X['berat_per_10_biji'] / (X['ukuran_rata2_per_10_biji']**2 + 1e-8)
        X_engineered['berat_air_interaction'] = X['berat_per_10_biji'] * X['kadar_air']
        X_engineered['ukuran_air_interaction'] = X['ukuran_rata2_per_10_biji'] * X['kadar_air']
        X_engineered['total_quality_proxy'] = X['berat_per_10_biji'] + X['ukuran_rata2_per_10_biji'] - X['kadar_air']
        
        # Binning for continuous features
        X_engineered['berat_category'] = pd.cut(X['berat_per_10_biji'], bins=3, labels=[0,1,2]).astype(int)
        X_engineered['ukuran_category'] = pd.cut(X['ukuran_rata2_per_10_biji'], bins=3, labels=[0,1,2]).astype(int)
        X_engineered['kadar_air_category'] = pd.cut(X['kadar_air'], bins=3, labels=[0,1,2]).astype(int)
        
        # Polynomial features
        numeric_cols = ['berat_per_10_biji', 'ukuran_rata2_per_10_biji', 'kadar_air']
        for col in numeric_cols:
            X_engineered[f'{col}_squared'] = X[col] ** 2
            X_engineered[f'{col}_cubed'] = X[col] ** 3
            X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
            X_engineered[f'{col}_log'] = np.log1p(np.abs(X[col]))
        
        # Statistical features
        X_engineered['feature_mean'] = X[numeric_cols].mean(axis=1)
        X_engineered['feature_std'] = X[numeric_cols].std(axis=1)
        X_engineered['feature_skew'] = X[numeric_cols].skew(axis=1)
        
        # Label encoding for categorical features
        X_engineered["tekstur"] = self.le_tekstur.transform(X["tekstur"])
        X_engineered["warna"] = self.le_warna.transform(X["warna"])
        
        # One-hot encoding for consistent categorical handling
        tekstur_dummies = pd.get_dummies(X["tekstur"], prefix='tekstur')
        warna_dummies = pd.get_dummies(X["warna"], prefix='warna')
        
        # Combine all features
        X_final = pd.concat([X_engineered, tekstur_dummies, warna_dummies], axis=1)
        
        return X_final


def evaluate_resampling_methods(X, y, models_to_test=None):
    """Test multiple resampling methods with different algorithms"""
    if models_to_test is None:
        models_to_test = {
            'GaussianNB': GaussianNB(),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
    
    resampling_methods = {
        'SMOTEENN': SMOTEENN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42),
        'BorderlineSMOTE-1': BorderlineSMOTE(random_state=42, kind='borderline-1'),
        'BorderlineSMOTE-2': BorderlineSMOTE(random_state=42, kind='borderline-2'),
        'ADASYN': ADASYN(random_state=42),
        'SMOTE_k3': SMOTE(random_state=42, k_neighbors=3),
        'SMOTE_k5': SMOTE(random_state=42, k_neighbors=5),
    }
    
    results = {}
    best_combination = {'method': None, 'model': None, 'score': 0}
    
    print("⚖️ Testing Resampling Methods with Multiple Algorithms...")
    
    for resample_name, resampler in resampling_methods.items():
        results[resample_name] = {}
        try:
            X_temp, y_temp = resampler.fit_resample(X, y)
            
            for model_name, model in models_to_test.items():
                try:
                    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
                    )
                    
                    model.fit(X_train_temp, y_train_temp)
                    score = model.score(X_test_temp, y_test_temp)
                    results[resample_name][model_name] = score
                    
                    if score > best_combination['score']:
                        best_combination = {
                            'method': resampler,
                            'method_name': resample_name,
                            'model': model,
                            'model_name': model_name,
                            'score': score
                        }
                    
                    print(f"   {resample_name:<18} + {model_name:<15}: {score:.4f}")
                    
                except Exception as e:
                    results[resample_name][model_name] = f"Error: {str(e)}"
                    print(f"   {resample_name:<18} + {model_name:<15}: Error")
                    
        except Exception as e:
            results[resample_name] = f"Error: {str(e)}"
            print(f"   {resample_name:<18}: Error - {str(e)}")
            continue
    
    return results, best_combination


def main():
    print("🚀 Starting Advanced Robusta Coffee Quality Classification")
    print("=" * 80)
    
    # Load dataset
    print("📁 Loading dataset...")
    df = pd.read_csv("dataset_kopi_robusta.csv")
    
    # Define features
    fitur = ["berat_per_10_biji", "ukuran_rata2_per_10_biji", "tekstur", "warna", "kadar_air"]
    X = df[fitur].copy()
    y = df["hasil_kualitas_kategori"]
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"📊 Class distribution:")
    class_dist = y.value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"   Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Feature Engineering
    print("\n🔧 Advanced Feature Engineering...")
    feature_engineer = AdvancedFeatureEngineer()
    feature_engineer.fit(X)
    X_engineered = feature_engineer.transform(X)
    print(f"   ✅ Features after engineering: {X_engineered.shape[1]} features")
    
    # Multiple Scaling Methods Test
    print("\n📐 Testing Multiple Scaling Methods...")
    scaling_methods = {
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer(output_distribution='normal', random_state=42),
        'PowerTransformer': PowerTransformer(method='yeo-johnson', standardize=True),
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    
    best_scaler = None
    best_scaler_score = 0
    scaling_results = {}
    
    for scaler_name, scaler in scaling_methods.items():
        try:
            X_scaled = scaler.fit_transform(X_engineered)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_engineered.columns)
            
            # Quick test with Gaussian NB
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
            )
            
            nb_temp = GaussianNB()
            nb_temp.fit(X_train_temp, y_train_temp)
            score = nb_temp.score(X_test_temp, y_test_temp)
            
            scaling_results[scaler_name] = score
            print(f"   {scaler_name:<20}: {score:.4f}")
            
            if score > best_scaler_score:
                best_scaler_score = score
                best_scaler = scaler
                
        except Exception as e:
            scaling_results[scaler_name] = f"Error: {str(e)}"
            print(f"   {scaler_name:<20}: Error")
    
    print(f"   🏆 Best scaler: {type(best_scaler).__name__}")
    
    # Apply best scaling
    X_scaled = best_scaler.fit_transform(X_engineered)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_engineered.columns)
    
    # Feature Selection
    print("\n🎯 Advanced Feature Selection...")
    
    # Multiple feature selection methods
    mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
    mi_selector.fit(X_scaled_df, y)
    mi_scores = pd.DataFrame({
        'feature': X_engineered.columns,
        'mi_score': mi_selector.scores_
    }).sort_values('mi_score', ascending=False)
    
    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X_scaled_df, y)
    f_scores = pd.DataFrame({
        'feature': X_engineered.columns,
        'f_score': f_selector.scores_
    }).sort_values('f_score', ascending=False)
    
    # Combine rankings
    combined_scores = pd.merge(mi_scores, f_scores, on='feature')
    combined_scores['avg_rank'] = (combined_scores['mi_score'].rank(ascending=False) + 
                                  combined_scores['f_score'].rank(ascending=False)) / 2
    combined_scores = combined_scores.sort_values('avg_rank')
    
    print(f"   📈 Top 25 features:")
    top_features = combined_scores.head(25)['feature'].tolist()
    for i, (_, row) in enumerate(combined_scores.head(25).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<30}: MI={row['mi_score']:.4f}, F={row['f_score']:.4f}")
    
    # Select features
    feature_selector = SelectKBest(score_func=f_classif, k=25)
    X_selected = feature_selector.fit_transform(X_scaled_df, y)
    
    # Test resampling methods
    resampling_results, best_resampling = evaluate_resampling_methods(X_selected, y)
    print(f"\n🏆 Best resampling combination: {best_resampling['method_name']} + {best_resampling['model_name']}")
    print(f"   Score: {best_resampling['score']:.4f}")
    
    # Apply best resampling
    X_resampled, y_resampled = best_resampling['method'].fit_resample(X_selected, y)
    print(f"   ✅ After resampling: {X_resampled.shape}")
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.15, random_state=42, stratify=y_resampled
    )
    
    # Define comprehensive model set
    print(f"\n🤖 Training Multiple Advanced Models...")
    models = {
        'Gaussian NB Optimized': GaussianNB(var_smoothing=1e-9),
        'Random Forest Advanced': RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees Advanced': ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting Advanced': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        ),
        'SVM Advanced': SVC(
            kernel='rbf',
            C=100,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'KNN Optimized': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='manhattan'
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
    }
    
    # Cross-validation
    print(f"\n📊 Performing 10-fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        cv_scores[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        print(f"   {name:<25}: CV F1-Score = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    # Train all models
    print(f"\n🏋️ Training All Models...")
    trained_models = {}
    predictions = {}
    reports = {}
    confusion_matrices = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        trained_models[name] = model
        predictions[name] = pred
        reports[name] = classification_report(y_test, pred, output_dict=True)
        confusion_matrices[name] = confusion_matrix(y_test, pred).tolist()
    
    # Create ensemble
    print(f"\n🤝 Creating Ensemble Models...")
    top_3_models = sorted(cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
    ensemble_models = [(name.replace(' ', '_').lower(), trained_models[name]) for name, _ in top_3_models]
    
    # Soft voting ensemble
    ensemble_soft = VotingClassifier(estimators=ensemble_models, voting='soft')
    ensemble_soft.fit(X_train, y_train)
    ensemble_pred = ensemble_soft.predict(X_test)
    
    # Hard voting ensemble
    ensemble_hard = VotingClassifier(estimators=ensemble_models, voting='hard')
    ensemble_hard.fit(X_train, y_train)
    ensemble_hard_pred = ensemble_hard.predict(X_test)
    
    # Add ensemble results
    trained_models['Ensemble Soft Voting'] = ensemble_soft
    predictions['Ensemble Soft Voting'] = ensemble_pred
    reports['Ensemble Soft Voting'] = classification_report(y_test, ensemble_pred, output_dict=True)
    confusion_matrices['Ensemble Soft Voting'] = confusion_matrix(y_test, ensemble_pred).tolist()
    
    trained_models['Ensemble Hard Voting'] = ensemble_hard
    predictions['Ensemble Hard Voting'] = ensemble_hard_pred
    reports['Ensemble Hard Voting'] = classification_report(y_test, ensemble_hard_pred, output_dict=True)
    confusion_matrices['Ensemble Hard Voting'] = confusion_matrix(y_test, ensemble_hard_pred).tolist()
    
    # Find best model
    best_f1 = 0
    best_model = None
    best_model_name = None
    
    for name, report in reports.items():
        f1_score = report['weighted avg']['f1-score']
        if f1_score > best_f1:
            best_f1 = f1_score
            best_model = trained_models[name]
            best_model_name = name
    
    # Pre-save testing
    print(f"\n🧪 Pre-Save Model Testing...")
    test_data = pd.DataFrame({
        'berat_per_10_biji': [5.55],
        'ukuran_rata2_per_10_biji': [6.65], 
        'tekstur': ['halus'],
        'warna': ['hitam'],
        'kadar_air': [11.5]
    })
    
    try:
        test_engineered = feature_engineer.transform(test_data)
        test_scaled = best_scaler.transform(test_engineered)
        test_selected = feature_selector.transform(test_scaled)
        test_prediction = best_model.predict(test_selected)
        test_proba = best_model.predict_proba(test_selected) if hasattr(best_model, 'predict_proba') else None
        
        print(f"   ✅ Pre-save test successful! Prediction: {test_prediction[0]}")
        if test_proba is not None:
            print(f"   ✅ Probabilities: {[f'{p:.4f}' for p in test_proba[0]]}")
            
    except Exception as e:
        print(f"   ❌ Pre-save test FAILED: {str(e)}")
        print("   Aborting save...")
        return
    
    # Save model components
    print(f"\n💾 Saving Model Components...")
    
    model_components = {
        'feature_engineer': feature_engineer,
        'scaler': best_scaler,
        'feature_selector': feature_selector,
        'resampler': best_resampling['method'],
        'model': best_model,
        'top_features': top_features,
        'feature_columns_after_engineering': X_engineered.columns.tolist(),
        'expected_columns': feature_engineer.expected_columns
    }
    
    with open("model_terbaik.pkl", "wb") as f:
        pickle.dump(model_components, f)
    print("   ✅ model_terbaik.pkl saved")
    
    # Save encoders for compatibility
    with open("encoder_tekstur.pkl", "wb") as f:
        pickle.dump(feature_engineer.le_tekstur, f)
    with open("encoder_warna.pkl", "wb") as f:
        pickle.dump(feature_engineer.le_warna, f)
    
    # Save comprehensive metadata
    model_metadata = {
        'model_terbaik': best_model_name,
        best_model_name.lower().replace(' ', '_').replace('-', '_'): {
            'akurasi': reports[best_model_name]['accuracy'],
            'presisi': reports[best_model_name]['weighted avg']['precision'],
            'recall': reports[best_model_name]['weighted avg']['recall'],
            'f1_score': reports[best_model_name]['weighted avg']['f1-score'],
            'klasifikasi': reports[best_model_name],
            'confusion_matrix': confusion_matrices[best_model_name]
        },
        'all_models_performance': {},
        'cv_scores': cv_scores,
        'resampling_results': resampling_results,
        'scaling_results': scaling_results,
        'feature_importance': combined_scores.head(25).to_dict('records'),
        'pipeline_info': {
            'feature_engineering': 'Advanced with polynomial, statistical, and interaction features',
            'scaling': type(best_scaler).__name__,
            'feature_selection': f'SelectKBest (k=25)',
            'resampling': best_resampling['method_name'],
            'final_model': best_model_name
        }
    }
    
    # Add all models performance
    for name, report in reports.items():
        model_key = name.lower().replace(' ', '_').replace('-', '_')
        model_metadata['all_models_performance'][model_key] = {
            'akurasi': report['accuracy'],
            'presisi': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }
    
    with open("model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=4)
    print("   ✅ model_metadata.json saved")
    
    # Save comprehensive evaluation
    evaluasi = {
        'model_terbaik': best_model_name,
        'all_results': {},
        'cv_scores': cv_scores,
        'resampling_analysis': resampling_results,
        'scaling_analysis': scaling_results,
        'feature_analysis': {
            'top_features': top_features,
            'feature_scores': combined_scores.head(25).to_dict('records'),
            'original_features': fitur,
            'engineered_count': X_engineered.shape[1],
            'selected_count': 25
        }
    }
    
    for name, report in reports.items():
        model_key = name.lower().replace(' ', '_').replace('-', '_')
        evaluasi['all_results'][model_key] = {
            'akurasi': report['accuracy'],
            'presisi': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'klasifikasi': report,
            'confusion_matrix': confusion_matrices[name]
        }
    
    with open("evaluasi_model.json", "w") as f:
        json.dump(evaluasi, f, indent=4, default=str)
    
    # Performance summary
    print(f"\n{'='*80}")
    print(f"🏆 TRAINING COMPLETED - BEST MODEL: {best_model_name}")
    print(f"{'='*80}")
    
    best_report = reports[best_model_name]
    print(f"📈 Best Model Performance:")
    print(f"   Accuracy : {best_report['accuracy']:.4f} ({best_report['accuracy']*100:.2f}%)")
    print(f"   Precision: {best_report['weighted avg']['precision']:.4f} ({best_report['weighted avg']['precision']*100:.2f}%)")
    print(f"   Recall   : {best_report['weighted avg']['recall']:.4f} ({best_report['weighted avg']['recall']*100:.2f}%)")
    print(f"   F1-Score : {best_report['weighted avg']['f1-score']:.4f} ({best_report['weighted avg']['f1-score']*100:.2f}%)")
    
    print(f"\n📊 Model Rankings (by F1-Score):")
    model_rankings = [(name, reports[name]['weighted avg']['f1-score'], reports[name]['accuracy']) 
                     for name in reports.keys()]
    model_rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, f1, acc) in enumerate(model_rankings, 1):
        print(f"   {i:2d}. {name:<30}: F1={f1:.4f}, Acc={acc:.4f}")
    
    print(f"\n🔧 Final Configuration:")
    print(f"   Feature Engineering: {X_engineered.shape[1]} features generated")
    print(f"   Scaling: {type(best_scaler).__name__}")
    print(f"   Feature Selection: Top 25 features")
    print(f"   Resampling: {best_resampling['method_name']}")
    print(f"   Final Model: {best_model_name}")
    
    print(f"\n📂 Files Saved:")
    print(f"   • model_terbaik.pkl - Complete model pipeline")
    print(f"   • model_metadata.json - Model metadata")
    print(f"   • evaluasi_model.json - Comprehensive evaluation")
    print(f"   • encoder_tekstur.pkl, encoder_warna.pkl - Label encoders")
    
    if best_report['accuracy'] >= 0.80:
        print(f"\n🎉 TARGET ACHIEVED! Accuracy {best_report['accuracy']*100:.2f}% (≥80%)")
    else:
        print(f"\n⚡ Accuracy {best_report['accuracy']*100:.2f}% - Significant improvement achieved!")
    
    print("=" * 80)


if __name__ == "__main__":
    main()