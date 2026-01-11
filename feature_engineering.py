import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from scipy import stats

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le_tekstur = LabelEncoder()
        self.le_warna = LabelEncoder()
        self.fitted = False
        # Store fitted PowerTransformers untuk digunakan di transform
        self.power_transformers = {}
        # Store statistics untuk normalization
        self.stats = {}
        # Store expected columns after one-hot encoding
        self.expected_columns = None

    def __setstate__(self, state):
        """Override untuk handle unpickle dari versi lama (backward compatibility)"""
        self.__dict__.update(state)
        
        # Inisialisasi attribute baru jika tidak ada (untuk model lama)
        if not hasattr(self, 'power_transformers'):
            self.power_transformers = {}
        if not hasattr(self, 'stats'):
            self.stats = {}
        if not hasattr(self, 'expected_columns'):
            self.expected_columns = None
        if not hasattr(self, 'fitted'):
            self.fitted = False

    def fit(self, X, y=None):
        # Fit label encoders
        self.le_tekstur.fit(X['tekstur'])
        self.le_warna.fit(X['warna'])
        
        # Fit PowerTransformers untuk setiap kolom numerik
        numeric_cols = ['berat_per_10_biji', 'ukuran_rata2_per_10_biji', 'kadar_air']
        for col in numeric_cols:
            try:
                pt = PowerTransformer(method='yeo-johnson')
                pt.fit(X[[col]])
                self.power_transformers[col] = pt
            except:
                # Jika gagal, gunakan transformasi sederhana
                self.power_transformers[col] = None
                
        # Store statistics untuk Z-score normalization
        self.stats = {
            'berat_per_10_biji_mean': X['berat_per_10_biji'].mean(),
            'berat_per_10_biji_std': X['berat_per_10_biji'].std(),
            'ukuran_rata2_per_10_biji_mean': X['ukuran_rata2_per_10_biji'].mean(),
            'ukuran_rata2_per_10_biji_std': X['ukuran_rata2_per_10_biji'].std(),
            'kadar_air_mean': X['kadar_air'].mean(),
            'kadar_air_std': X['kadar_air'].std(),
            'berat_per_10_biji_min': X['berat_per_10_biji'].min(),
            'berat_per_10_biji_max': X['berat_per_10_biji'].max(),
            'ukuran_rata2_per_10_biji_min': X['ukuran_rata2_per_10_biji'].min(),
            'ukuran_rata2_per_10_biji_max': X['ukuran_rata2_per_10_biji'].max(),
            'kadar_air_min': X['kadar_air'].min(),
            'kadar_air_max': X['kadar_air'].max(),
        }
        
        # Fit untuk mendapatkan expected columns setelah one-hot encoding
        X_temp = self._apply_feature_engineering(X, is_fitting=True)
        self.expected_columns = list(X_temp.columns)
        
        self.fitted = True
        return self
        
    def transform(self, X):
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        X_engineered = self._apply_feature_engineering(X, is_fitting=False)
        
        # Pastikan kolom sesuai dengan yang diharapkan
        X_engineered = self._align_columns(X_engineered)
        
        return X_engineered
    
    def _apply_feature_engineering(self, X, is_fitting=False):
        """
        Apply all feature engineering steps
        """
        X_engineered = X.copy()
        
        # 1. LOG TRANSFORMATIONS - Lebih aman untuk single prediction
        numeric_cols = ['berat_per_10_biji', 'ukuran_rata2_per_10_biji', 'kadar_air']
        for col in numeric_cols:
            # Log transform untuk mendekati distribusi normal
            X_engineered[f'{col}_log'] = np.log1p(X[col])
            X_engineered[f'{col}_log2'] = np.log2(X[col] + 1)
            X_engineered[f'{col}_log10'] = np.log10(X[col] + 1)
            
            # Yeo-Johnson transformation - gunakan fitted transformer atau fallback
            if hasattr(self, 'power_transformers') and self.power_transformers.get(col) is not None:
                try:
                    X_engineered[f'{col}_yeojohnson'] = self.power_transformers[col].transform(X[[col]]).flatten()
                except:
                    # Fallback ke log transformation jika gagal
                    X_engineered[f'{col}_yeojohnson'] = np.log1p(X[col])
            else:
                # Fallback untuk model lama
                X_engineered[f'{col}_yeojohnson'] = np.log1p(X[col])

        # 2. RATIO FEATURES - Naive Bayes bagus dengan rasio
        X_engineered['berat_ukuran_ratio'] = X['berat_per_10_biji'] / (X['ukuran_rata2_per_10_biji'] + 1e-8)
        X_engineered['berat_air_ratio'] = X['berat_per_10_biji'] / (X['kadar_air'] + 1e-8)
        X_engineered['ukuran_air_ratio'] = X['ukuran_rata2_per_10_biji'] / (X['kadar_air'] + 1e-8)

        # 3. STATISTICAL FEATURES
        X_engineered['density_proxy'] = X['berat_per_10_biji'] / (X['ukuran_rata2_per_10_biji']**2 + 1e-8)
        X_engineered['volume_proxy'] = X['ukuran_rata2_per_10_biji']**3
        X_engineered['surface_area_proxy'] = X['ukuran_rata2_per_10_biji']**2

        # 4. BINNING - Menggunakan rule-based binning untuk single prediction
        for col in numeric_cols:
            if len(X) == 1:  # Single prediction
                val = X[col].iloc[0]
                # Simple rule-based binning berdasarkan domain knowledge
                if col == 'berat_per_10_biji':
                    X_engineered[f'{col}_q3'] = [0 if val < 5.5 else 1 if val < 6.5 else 2]
                    X_engineered[f'{col}_q5'] = [min(4, max(0, int((val - 4) / 0.8)))]
                    X_engineered[f'{col}_q7'] = [min(6, max(0, int((val - 4) / 0.6)))]
                elif col == 'ukuran_rata2_per_10_biji':
                    X_engineered[f'{col}_q3'] = [0 if val < 7.5 else 1 if val < 9 else 2]
                    X_engineered[f'{col}_q5'] = [min(4, max(0, int((val - 6) / 1.5)))]
                    X_engineered[f'{col}_q7'] = [min(6, max(0, int((val - 6) / 1.0)))]
                else:  # kadar_air
                    X_engineered[f'{col}_q3'] = [0 if val < 11 else 1 if val < 12.5 else 2]
                    X_engineered[f'{col}_q5'] = [min(4, max(0, int((val - 10) / 1.0)))]
                    X_engineered[f'{col}_q7'] = [min(6, max(0, int((val - 10) / 0.7)))]
                
                # Equal-width binning
                X_engineered[f'{col}_bin3'] = X_engineered[f'{col}_q3']
                X_engineered[f'{col}_bin5'] = X_engineered[f'{col}_q5']
            else:
                # Batch prediction - gunakan quantile binning
                try:
                    X_engineered[f'{col}_q3'] = pd.qcut(X[col], q=3, labels=[0,1,2], duplicates='drop').astype(float)
                    X_engineered[f'{col}_q5'] = pd.qcut(X[col], q=5, labels=[0,1,2,3,4], duplicates='drop').astype(float)
                    X_engineered[f'{col}_q7'] = pd.qcut(X[col], q=7, labels=[0,1,2,3,4,5,6], duplicates='drop').astype(float)
                except:
                    # Fallback if quantile binning fails
                    X_engineered[f'{col}_q3'] = pd.cut(X[col], bins=3, labels=[0,1,2]).astype(float)
                    X_engineered[f'{col}_q5'] = pd.cut(X[col], bins=5, labels=[0,1,2,3,4]).astype(float)
                    X_engineered[f'{col}_q7'] = pd.cut(X[col], bins=7, labels=[0,1,2,3,4,5,6]).astype(float)
                
                # Equal-width binning
                X_engineered[f'{col}_bin3'] = pd.cut(X[col], bins=3, labels=[0,1,2]).astype(float)
                X_engineered[f'{col}_bin5'] = pd.cut(X[col], bins=5, labels=[0,1,2,3,4]).astype(float)

        # 5. Z-SCORE NORMALIZATION - gunakan statistics yang disimpan saat fit
        for col in numeric_cols:
            mean_key = f'{col}_mean'
            std_key = f'{col}_std'
            min_key = f'{col}_min'
            max_key = f'{col}_max'
            
            if hasattr(self, 'stats') and mean_key in self.stats and std_key in self.stats:
                X_engineered[f'{col}_zscore'] = (X[col] - self.stats[mean_key]) / self.stats[std_key] if self.stats[std_key] != 0 else 0.0
            else:
                X_engineered[f'{col}_zscore'] = 0.0  # Fallback untuk model lama
            
            if hasattr(self, 'stats') and min_key in self.stats and max_key in self.stats:
                X_engineered[f'{col}_minmax'] = (X[col] - self.stats[min_key]) / (self.stats[max_key] - self.stats[min_key]) if (self.stats[max_key] - self.stats[min_key]) != 0 else 0.0
            else:
                X_engineered[f'{col}_minmax'] = 0.0  # Fallback untuk model lama

        # 6. POLYNOMIAL FEATURES dengan degree rendah
        for col in numeric_cols:
            X_engineered[f'{col}_squared'] = X[col] ** 2
            X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))

        # 7. INTERACTION FEATURES
        X_engineered['berat_air_interaction'] = X['berat_per_10_biji'] * X['kadar_air']
        X_engineered['ukuran_air_interaction'] = X['ukuran_rata2_per_10_biji'] * X['kadar_air']
        X_engineered['all_numeric_product'] = X['berat_per_10_biji'] * X['ukuran_rata2_per_10_biji'] * X['kadar_air']

        # Label encoding untuk categorical (temporary)
        X_temp = X_engineered.copy()
        X_temp["tekstur_encoded"] = self.le_tekstur.transform(X["tekstur"])
        X_temp["warna_encoded"] = self.le_warna.transform(X["warna"])

        # One-hot encoding untuk Naive Bayes dengan konsistensi
        if is_fitting:
            # Saat fitting, buat semua kolom dummy yang mungkin
            tekstur_dummies = pd.get_dummies(X_temp['tekstur_encoded'], prefix='tekstur')
            warna_dummies = pd.get_dummies(X_temp['warna_encoded'], prefix='warna')
        else:
            # Saat transform, buat dummy dengan konsistensi
            # Get all possible values from fitted encoders
            all_tekstur_values = range(len(self.le_tekstur.classes_))
            all_warna_values = range(len(self.le_warna.classes_))
            
            # Create dummies for tekstur
            tekstur_dummies = pd.DataFrame(0, index=X_temp.index, 
                                        columns=[f'tekstur_{i}' for i in all_tekstur_values])
            for i, val in enumerate(X_temp['tekstur_encoded']):
                tekstur_dummies.loc[tekstur_dummies.index[i], f'tekstur_{val}'] = 1
                
            # Create dummies for warna  
            warna_dummies = pd.DataFrame(0, index=X_temp.index,
                                    columns=[f'warna_{i}' for i in all_warna_values])
            for i, val in enumerate(X_temp['warna_encoded']):
                warna_dummies.loc[warna_dummies.index[i], f'warna_{val}'] = 1

        # Remove original categorical columns and temporary encoded columns
        X_engineered = X_engineered.drop(columns=['tekstur', 'warna'])
        
        # Add dummy columns
        X_engineered = pd.concat([X_engineered, tekstur_dummies, warna_dummies], axis=1)
        
        # Fill NaN values
        X_engineered = X_engineered.fillna(0)
        
        return X_engineered
    
    def _align_columns(self, X_engineered):
        """
        Pastikan kolom sesuai dengan yang diharapkan saat training
        """
        if self.expected_columns is None:
            return X_engineered
        
        # Add missing columns with 0 values
        for col in self.expected_columns:
            if col not in X_engineered.columns:
                X_engineered[col] = 0
        
        # Remove extra columns that weren't in training
        extra_cols = [col for col in X_engineered.columns if col not in self.expected_columns]
        if extra_cols:
            X_engineered = X_engineered.drop(columns=extra_cols)
        
        # Reorder columns to match training order
        X_engineered = X_engineered[self.expected_columns]
        
        return X_engineered
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Alias untuk kompatibilitas dengan model lama
class AdvancedFeatureEngineer(FeatureEngineer):
    """
    Alias dari FeatureEngineer untuk mendukung loading model lama yang
    disimpan dengan nama class AdvancedFeatureEngineer.
    Tidak menambah atau mengubah fungsi apapun.
    """
    pass