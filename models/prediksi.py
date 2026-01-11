from . import db
from datetime import datetime

class RiwayatPrediksi(db.Model):
    __tablename__ = 'riwayat_prediksi'

    id = db.Column(db.Integer, primary_key=True)
    berat = db.Column(db.Float)
    ukuran = db.Column(db.Float)
    kadar_air = db.Column(db.Float)
    tekstur = db.Column(db.String)
    warna = db.Column(db.String)
    harga = db.Column(db.Float)
    tempat_produksi = db.Column(db.String)
    nama_penguji = db.Column(db.String)
    hasil = db.Column(db.String)
    tanggal_prediksi = db.Column(db.DateTime, default=datetime.utcnow)
