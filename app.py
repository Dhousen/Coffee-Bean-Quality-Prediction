# ==== PATCH UNTUK MODEL LAMA ====
import joblib
import sys
import types
from feature_engineering import FeatureEngineer

# Buat module dummy "__main__" kalau belum ada
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")

# Aliaskan AdvancedFeatureEngineer -> FeatureEngineer
setattr(sys.modules["__main__"], "AdvancedFeatureEngineer", FeatureEngineer)

# Fungsi load model & preprocessor
def load_model_components(model_path, preprocessor_path):
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model atau preprocessor: {e}")
# ==== END PATCH ====

from flask import Flask, render_template, request, redirect, url_for, flash, session as flask_session, send_file, Blueprint
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app
from models import db
from models.user import User
from models.prediksi import RiwayatPrediksi
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import io
import os
import joblib
import json
from xhtml2pdf import pisa
from feature_engineering import FeatureEngineer
# -------------------- Konfigurasi Aplikasi Flask --------------------
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(basedir, 'instance', 'your_database.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'a9f1c8fbb1f94e58a3e0a962dab8b5cf'

# -------------------- Inisialisasi Ekstensi --------------------
db.init_app(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------- Routing --------------------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan.', 'error')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='sha256')
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()

        flash('Registrasi berhasil!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('home'))
        flash('Username atau password salah.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Silakan login kembali.', 'info')
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/form')
@login_required
def form_input_page():
    return render_template('form_input.html')

# -------------------- Blueprint Prediksi --------------------
predict_bp = Blueprint('predict', __name__)

# Load model dan semua komponen preprocessing
# Load model dan semua komponen preprocessing
try:
    with open('model_terbaik.pkl', 'rb') as f:
        model_components = pickle.load(f)

    feature_engineer = model_components.get('feature_engineer')
    # Gunakan scaler dari model_components
    best_scaler = model_components.get('scaler')  # Konsisten dengan train_model.py
    feature_selector = model_components.get('feature_selector')
    resampler = model_components.get('resampler')
    model = model_components.get('model')

    if best_scaler is None:
        raise ValueError("Scaler tidak ditemukan di model_components")

    print("✅ Model dan pipeline berhasil dimuat!")
    print(f"   - Feature Engineer: {type(feature_engineer).__name__ if feature_engineer else 'None'}")
    print(f"   - Scaler: {type(best_scaler).__name__ if best_scaler else 'None'}")
    print(f"   - Feature Selector: {type(feature_selector).__name__ if feature_selector else 'None'}")
    print(f"   - Resampler: {type(resampler).__name__ if resampler else 'None'}")
    print(f"   - Model: {type(model).__name__ if model else 'None'}")

except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    # Fallback untuk backward compatibility
    try:
        model = joblib.load('model_terbaik.pkl')
        encoder_tekstur = joblib.load('encoder_tekstur.pkl')
        encoder_warna = joblib.load('encoder_warna.pkl')
        print("⚠️ Menggunakan model lama tanpa pipeline lengkap")
        model_components = None
        best_scaler = None  # Tambahkan ini untuk mencegah NameError
    except Exception as e2:
        print(f"❌ Error loading fallback model: {str(e2)}")
        model_components = None
        model = None
        best_scaler = None  # Tambahkan ini untuk mencegah NameError

def preprocess_single_prediction(berat, ukuran, tekstur, warna, kadar_air):
    """
    Mengembalikan array yang sudah melalui feature_engineer -> scaler -> feature_selector
    Pastikan nama kolom input sama seperti saat training:
      'berat_per_10_biji', 'ukuran_rata2_per_10_biji', 'tekstur', 'warna', 'kadar_air'
    """
    global feature_engineer, best_scaler, feature_selector

    if feature_engineer is None:
        raise ValueError("Feature engineer belum tersedia")
    if best_scaler is None:
        raise ValueError("Scaler belum tersedia")
    if feature_selector is None:
        raise ValueError("Feature selector belum tersedia")

    # Buat DataFrame input (1 baris)
    df = pd.DataFrame([{
        'berat_per_10_biji': float(berat),
        'ukuran_rata2_per_10_biji': float(ukuran),
        'tekstur': str(tekstur),
        'warna': str(warna),
        'kadar_air': float(kadar_air)
    }])

    # Transform feature engineering
    engineered = feature_engineer.transform(df)  # returns DataFrame
    # Pastikan kolom urut sesuai
    engineered_df = pd.DataFrame(engineered, columns=feature_engineer.expected_columns) if not isinstance(engineered, pd.DataFrame) else engineered

    # Scale
    scaled = best_scaler.transform(engineered_df)  # numpy array

    # Feature selector (SelectKBest) -> hasil array
    selected = feature_selector.transform(scaled)

    return selected  # numpy array siap untuk predict

@predict_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Ambil data dari form
            berat = float(request.form['berat'])
            ukuran = float(request.form['ukuran'])
            tekstur = request.form['tekstur']
            warna = request.form['warna']
            kadar_air = float(request.form['kadar_air'])
            tempat_produksi = request.form.get('tempat_produksi', 'Tidak Diketahui')
            nama_penguji = request.form.get('nama_penguji', 'Tidak Diketahui')
            harga = float(request.form.get('harga', 0))

            # pastikan model sudah ter-load
            global model, model_components, encoder_tekstur, encoder_warna

            if model is None and model_components is None:
                # coba reload sekali lagi
                load_model_components()
            
            if model is None:
                raise ValueError("Model belum dimuat di server. Pastikan file model_terbaik.pkl ada dan aplikasi sudah restart setelah menaruh file.")

            # Gunakan pipeline lengkap
            input_data = preprocess_single_prediction(berat, ukuran, tekstur, warna, kadar_air)

            hasil = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0].tolist() if hasattr(model, 'predict_proba') else None
            label_kelas = model.classes_.tolist() if hasattr(model, 'classes_') else None

            print(f"✅ Prediksi berhasil dengan pipeline lengkap")
            print(f"   Input shape setelah preprocessing: {input_data.shape}")
            print(f"   Hasil prediksi: {hasil}")

            # mapping hasil dsb (sama dengan yang kamu punya)
            label_mapping = { 0:'sangat buruk',1:'buruk',2:'standar',3:'baik',4:'sangat baik' }
            hasil_label = label_mapping.get(hasil, str(hasil))

            # Simpan ke DB (sama seperti kode awal)
            riwayat = RiwayatPrediksi(
                berat=berat,
                ukuran=ukuran,
                tekstur=tekstur,
                warna=warna,
                kadar_air=kadar_air,
                hasil=hasil_label,
                tempat_produksi=tempat_produksi,
                nama_penguji=nama_penguji,
                harga=harga
            )
            db.session.add(riwayat)
            db.session.commit()

            return render_template('hasil_prediksi.html',
                                   hasil=hasil_label,
                                   hasil_numeric=hasil,
                                   probabilities=probabilities,
                                   labels=[label_mapping.get(i, str(i)) for i in (label_kelas if label_kelas is not None else [])])

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Terjadi kesalahan: {str(e)}"

    return render_template('form_input.html')


@predict_bp.route('/test-model', methods=['GET'])
@login_required
def test_model():
    try:
        test_data = {'berat': 5.55, 'ukuran':6.65, 'tekstur':'halus', 'warna':'hitam', 'kadar_air':11.5}
        global model
        if model is None:
            # coba reload
            load_model_components()

        if model is None:
            return {'status':'error', 'message':'Model tidak tersedia. Periksa log server apakah model_terbaik.pkl berhasil dimuat.'}

        input_data = preprocess_single_prediction(test_data['berat'], test_data['ukuran'], test_data['tekstur'], test_data['warna'], test_data['kadar_air'])
        hasil = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0].tolist() if hasattr(model,'predict_proba') else None

        return {
            'status': 'success',
            'input': test_data,
            'processed_shape': input_data.shape,
            'hasil': int(hasil),
            'probabilities': probabilities,
            'model_type': type(model).__name__
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status':'error', 'message': str(e)}

# -------------------- Riwayat --------------------
@app.route('/history')
@login_required
def history():
    riwayat = RiwayatPrediksi.query.order_by(RiwayatPrediksi.tanggal_prediksi.desc()).all()
    return render_template('history.html', history=riwayat)

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_history(id):
    riwayat = RiwayatPrediksi.query.get_or_404(id)
    if request.method == 'POST':
        riwayat.nama_penguji = request.form['nama_penguji']
        riwayat.tempat_produksi = request.form['tempat_produksi']
        riwayat.harga = float(request.form['harga'])
        db.session.commit()
        flash('Data berhasil diperbarui.', 'success')
        return redirect(url_for('history'))
    return render_template('edit_history.html', riwayat=riwayat)


@app.route('/history/delete/<int:id>', methods=['POST'])
@login_required
def delete_history(id):
    riwayat = RiwayatPrediksi.query.get_or_404(id)
    db.session.delete(riwayat)
    db.session.commit()
    return redirect(url_for('history'))

@app.route('/delete_all_history', methods=['POST'])
@login_required
def delete_all_history():
    RiwayatPrediksi.query.delete()
    db.session.commit()
    flash('Semua riwayat berhasil dihapus.', 'success')
    return redirect(url_for('history'))

# -------------------- Export --------------------
@app.route('/export_excel')
@login_required
def export_excel():
    riwayat = RiwayatPrediksi.query.all()
    df = pd.DataFrame([{**r.__dict__} for r in riwayat])
    df.drop(columns=['_sa_instance_state'], inplace=True)
    df['tanggal_prediksi'] = df['tanggal_prediksi'].dt.strftime('%d-%m-%Y %H:%M')
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return send_file(output, download_name='riwayat_prediksi.xlsx', as_attachment=True)

@app.route('/export_pdf')
@login_required
def export_pdf():
    history = RiwayatPrediksi.query.all()

    basedir = os.path.abspath(os.path.dirname(__file__))
    logo_path = os.path.join(basedir, 'static', 'img', 'logo.png')

    # Render HTML dengan logo_path
    html = render_template('export_template.html', history=history, logo_path=logo_path)

    # Buat PDF
    pdf = io.BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf)

    if pisa_status.err:
        return 'Gagal membuat PDF.'

    pdf.seek(0)
    return send_file(pdf, download_name='riwayat_prediksi.pdf', as_attachment=True)

# -------------------- Fitur Pencarian --------------------
@app.route('/cari_kopi')
@login_required
def cari_kopi():
    return render_template('cari_kopi.html')

@app.route('/hasil_cari', methods=['POST'])
@login_required
def hasil_cari():
    print("Form data masuk:", request.form)  # ✅ Debug input

    def parse_float(val): 
        try: return float(val)
        except (ValueError, TypeError): return None

    query = RiwayatPrediksi.query

    if (val := parse_float(request.form.get('harga'))):
        print(f"Filter harga <= {val}")
        query = query.filter(RiwayatPrediksi.harga <= val)
    if (val := request.form.get('kualitas')):
        print(f"Filter kualitas == {val.strip()}")
        query = query.filter(RiwayatPrediksi.hasil.ilike(f"%{val.strip()}%"))
    if (val := parse_float(request.form.get('berat'))):
        query = query.filter(RiwayatPrediksi.berat <= val)
    if (val := parse_float(request.form.get('ukuran'))):
        query = query.filter(RiwayatPrediksi.ukuran <= val)
    if (val := request.form.get('tekstur')):
        query = query.filter(RiwayatPrediksi.tekstur.ilike(f"%{val.strip()}%"))
    if (val := request.form.get('warna')):
        query = query.filter(RiwayatPrediksi.warna.ilike(f"%{val.strip()}%"))
    if (val := parse_float(request.form.get('kadar_air'))):
        query = query.filter(RiwayatPrediksi.kadar_air <= val)

    hasil = query.all()
    print("Jumlah hasil ditemukan:", len(hasil))  # ✅ Debug hasil
    return render_template('hasil_cari.html', hasil_pencarian=hasil)

# -------------------- Evaluasi Model --------------------
evaluasi_bp = Blueprint('evaluasi_model', __name__)

@evaluasi_bp.route('/evaluasi')
@login_required
def evaluasi_model():
    try:
        with open('evaluasi_model.json') as f:
            data = json.load(f)

        # model_terbaik (string)
        model_terbaik = data.get('model_terbaik') or data.get('model_terbaik', None)
        if not model_terbaik:
            raise ValueError("Field 'model_terbaik' tidak ditemukan di evaluasi_model.json")

        # bentuk kunci yang dicari (dukungan beberapa format)
        model_key_variant1 = model_terbaik.lower().replace(" ", "_")
        model_key_variant2 = model_terbaik.lower().replace(" ", "_").replace("-", "_")

        # 1) coba format lama: top-level key sama dengan model_key
        hasil = data.get(model_key_variant1) or data.get(model_key_variant2)

        # 2) jika tidak ada, coba format baru: semua hasil berada di data['all_results']
        if not hasil:
            all_results = data.get('all_results', {})
            hasil = all_results.get(model_key_variant1) or all_results.get(model_key_variant2, {})

        if not hasil:
            raise ValueError("Hasil evaluasi tidak ditemukan untuk model terbaik.")

        # mengambil metrik
        akurasi = hasil.get('akurasi', 0) * 100 if isinstance(hasil.get('akurasi', 0), (int, float)) else 0
        klasifikasi = hasil.get('klasifikasi') or hasil.get('klasifikasi')
        if not klasifikasi:
            # kadang 'klasifikasi' sebenarnya berisi seluruh 'klasifikasi' di dalam 'klasifikasi' key,
            # tetapi jika tidak tersedia coba ambil yang ada di 'klasifikasi' nested_dictionary
            klasifikasi = hasil if isinstance(hasil, dict) else {}

        # jika klasifikasi berupa dict yang cocok, ubah menjadi dataframe
        df_klasifikasi = pd.DataFrame(klasifikasi).transpose().round(2)
        klasifikasi_html = df_klasifikasi.to_html(classes="table table-bordered table-striped text-sm", border=0)

        presisi = df_klasifikasi.loc['weighted avg','precision'] * 100 if 'weighted avg' in df_klasifikasi.index else 0
        recall = df_klasifikasi.loc['weighted avg','recall'] * 100 if 'weighted avg' in df_klasifikasi.index else 0
        f1 = df_klasifikasi.loc['weighted avg','f1-score'] * 100 if 'weighted avg' in df_klasifikasi.index else 0

        return render_template("evaluasi_model.html",
                               model=model_terbaik,
                               akurasi=f"{akurasi:.2f}",
                               presisi=f"{presisi:.2f}",
                               recall=f"{recall:.2f}",
                               f1=f"{f1:.2f}",
                               klasifikasi_dict=df_klasifikasi.to_dict(orient='index'),
                               klasifikasi_tabel=klasifikasi_html,
                               confusion_matrix=hasil.get("confusion_matrix"))

    except FileNotFoundError:
        return "File evaluasi_model.json tidak ditemukan."
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"


# -------------------- Main Runner --------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.register_blueprint(predict_bp)
    app.register_blueprint(evaluasi_bp)
    app.run(debug=True)
