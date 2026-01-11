from models.base import Base, engine
from models.prediksi import RiwayatPrediksi

# Membuat tabel di database
Base.metadata.create_all(bind=engine)
print("Database dan tabel berhasil dibuat.")
