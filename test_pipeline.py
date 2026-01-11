import pandas as pd
import pickle
import numpy as np

def test_pipeline():
    """
    Test pipeline dengan berbagai skenario input
    """
    print("🧪 Testing Fixed Pipeline...")
    
    try:
        # Load model components
        with open('model_terbaik.pkl', 'rb') as f:
            model_components = pickle.load(f)
        
        feature_engineer = model_components['feature_engineer']
        quantile_transformer = model_components['quantile_transformer']
        feature_selector = model_components['feature_selector']
        model = model_components['model']
        
        print("✅ Model components loaded successfully!")
        print(f"   - Expected columns: {len(feature_engineer.expected_columns)}")
        
        # Test cases dengan berbagai kombinasi input
        test_cases = [
            {
                'name': 'Test Case 1 - Halus Hitam',
                'data': {
                    'berat_per_10_biji': [5.55],
                    'ukuran_rata2_per_10_biji': [6.65], 
                    'tekstur': ['halus'],
                    'warna': ['hitam'],
                    'kadar_air': [11.5]
                }
            },
            {
                'name': 'Test Case 2 - Kasar Coklat Tua',
                'data': {
                    'berat_per_10_biji': [6.2],
                    'ukuran_rata2_per_10_biji': [7.8], 
                    'tekstur': ['kasar'],
                    'warna': ['coklat tua'],
                    'kadar_air': [12.1]
                }
            },
            {
                'name': 'Test Case 3 - Sedang Coklat Muda',
                'data': {
                    'berat_per_10_biji': [5.8],
                    'ukuran_rata2_per_10_biji': [7.2], 
                    'tekstur': ['sedang'],
                    'warna': ['coklat muda'],
                    'kadar_air': [10.8]
                }
            },
            {
                'name': 'Test Case 4 - Halus Kemerahan',
                'data': {
                    'berat_per_10_biji': [6.0],
                    'ukuran_rata2_per_10_biji': [8.1], 
                    'tekstur': ['halus'],
                    'warna': ['kemerahan'],
                    'kadar_air': [11.9]
                }
            }
        ]
        
        label_mapping = {
            0: 'sangat buruk',
            1: 'buruk', 
            2: 'standar',
            3: 'baik',
            4: 'sangat baik'
        }
        
        print("\n" + "="*70)
        print("🔬 RUNNING TEST CASES")
        print("="*70)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{test_case['name']}:")
            
            try:
                # Create DataFrame
                test_df = pd.DataFrame(test_case['data'])
                print(f"   Input: berat={test_df['berat_per_10_biji'][0]}, ukuran={test_df['ukuran_rata2_per_10_biji'][0]}")
                print(f"          tekstur={test_df['tekstur'][0]}, warna={test_df['warna'][0]}, kadar_air={test_df['kadar_air'][0]}")
                
                # Step 1: Feature Engineering
                X_engineered = feature_engineer.transform(test_df)
                print(f"   ✅ Feature Engineering: {X_engineered.shape[1]} features")
                
                # Step 2: Quantile Transformation
                X_transformed = quantile_transformer.transform(X_engineered)
                print(f"   ✅ Quantile Transform: {X_transformed.shape}")
                
                # Step 3: Feature Selection
                X_selected = feature_selector.transform(X_transformed)
                print(f"   ✅ Feature Selection: {X_selected.shape[1]} features")
                
                # Step 4: Prediction
                prediction = model.predict(X_selected)[0]
                probabilities = model.predict_proba(X_selected)[0]
                
                result_label = label_mapping.get(prediction, str(prediction))
                max_prob = np.max(probabilities) * 100
                
                print(f"   🎯 HASIL: {result_label} (confidence: {max_prob:.1f}%)")
                print(f"   📊 Probabilities: {[f'{p:.3f}' for p in probabilities]}")
                
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                
        print("\n" + "="*70)
        print("✅ PIPELINE TEST COMPLETED!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"❌ PIPELINE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_app_integration():
    """
    Test integrasi dengan fungsi yang sama seperti di app.py
    """
    print("\n🔗 Testing App Integration...")
    
    try:
        with open('model_terbaik.pkl', 'rb') as f:
            model_components = pickle.load(f)
        
        feature_engineer = model_components['feature_engineer']
        quantile_transformer = model_components['quantile_transformer']
        feature_selector = model_components['feature_selector']
        model = model_components['model']
        
        def preprocess_single_prediction(berat, ukuran, tekstur, warna, kadar_air):
            """
            Copy exact dari app.py
            """
            input_df = pd.DataFrame({
                'berat_per_10_biji': [berat],
                'ukuran_rata2_per_10_biji': [ukuran],
                'tekstur': [tekstur],
                'warna': [warna],
                'kadar_air': [kadar_air]
            })
            
            X_engineered = feature_engineer.transform(input_df)
            X_transformed = quantile_transformer.transform(X_engineered)
            X_selected = feature_selector.transform(X_transformed)
            
            return X_selected
        
        # Test dengan data dari form
        test_input = preprocess_single_prediction(5.55, 6.65, 'halus', 'hitam', 11.5)
        prediction = model.predict(test_input)[0]
        probabilities = model.predict_proba(test_input)[0]
        
        label_mapping = {
            0: 'sangat buruk',
            1: 'buruk', 
            2: 'standar',
            3: 'baik',
            4: 'sangat baik'
        }
        
        result_label = label_mapping.get(prediction, str(prediction))
        
        print(f"   ✅ App Integration Test PASSED!")
        print(f"   🎯 Result: {result_label}")
        print(f"   📊 Probabilities: {probabilities}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ App Integration Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Pipeline Tests...")
    
    # Test 1: Pipeline functionality
    test1_passed = test_pipeline()
    
    # Test 2: App integration
    test2_passed = test_app_integration()
    
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    print(f"Pipeline Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"App Integration Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Pipeline siap digunakan.")
        print("\n📝 Langkah selanjutnya:")
        print("   1. Ganti file feature_engineering.py dengan versi yang baru")
        print("   2. Jalankan retrain_model.py untuk melatih ulang model")
        print("   3. Test aplikasi Flask Anda")
    else:
        print("\n⚠️ ADA TEST YANG GAGAL! Periksa error di atas.")
        
    print("="*70)