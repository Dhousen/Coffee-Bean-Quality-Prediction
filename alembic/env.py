import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging.config import fileConfig

from sqlalchemy import create_engine
from alembic import context

# Import model dan db kamu, sesuaikan dengan struktur aplikasi kamu
from models.prediksi import RiwayatPrediksi, User  # Gantilah dengan lokasi model yang benar
from models.base import Base  # Pastikan Base adalah tempat metadata untuk semua model

# Ini adalah metadata yang digunakan untuk auto-generate migrasi
target_metadata = Base.metadata  # Gunakan Base metadata agar mencakup semua model

# Configure your connection string here
config = context.config
fileConfig(config.config_file_name)

# Menyiapkan koneksi database
def get_url():
    return "sqlite:///your_database.db"  # Gantilah dengan URL database yang benar

config.set_section_option('alembic', 'sqlalchemy.url', get_url())

def run_migrations_online():
    # Membuat engine SQLAlchemy
    engine = create_engine(get_url())
    connection = engine.connect()
    context.configure(
        connection=connection,
        target_metadata=target_metadata
    )

    try:
        with context.begin_transaction():
            context.run_migrations()
    finally:
        connection.close()

if context.is_offline_mode():
    print("Offline mode")
else:
    run_migrations_online()
