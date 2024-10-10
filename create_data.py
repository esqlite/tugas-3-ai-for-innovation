import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Menghasilkan data untuk kolom yang telah ditentukan
num_rows_2024 = 20  # Jumlah baris untuk tahun 2024
num_rows_2023 = 50  # Jumlah baris untuk tahun 2023
num_rows_2022 = 50  # Jumlah baris untuk tahun 2022
total_rows = num_rows_2024 + num_rows_2023 + num_rows_2022

# Menghasilkan tanggal untuk created_at dan updated_at
def generate_dates_with_time(num_rows, year):
    start_date = datetime(year, 1, 1)
    date_range = [start_date + timedelta(days=i) for i in range(365)]  # 1 tahun
    created_at_dates = []
    updated_at_dates = []
    for _ in range(num_rows):
        created_at = random.choice(date_range)
        created_at_dates.append(created_at.replace(hour=random.randint(0, 23), 
                                                    minute=random.randint(0, 59)))  # Menambahkan waktu ke created_at
        updated_at = created_at + timedelta(days=random.randint(1, 10), 
                                             hours=random.randint(0, 23), 
                                             minutes=random.randint(0, 59))
        updated_at_dates.append(updated_at)
    return created_at_dates, updated_at_dates

# Menghasilkan data untuk setiap tahun
created_at_2024, updated_at_2024 = generate_dates_with_time(num_rows_2024, 2024)
created_at_2023, updated_at_2023 = generate_dates_with_time(num_rows_2023, 2023)
created_at_2022, updated_at_2022 = generate_dates_with_time(num_rows_2022, 2022)

# Menggabungkan data
created_at_dates = np.concatenate([created_at_2024, created_at_2023, created_at_2022])
updated_at_dates = np.concatenate([updated_at_2024, updated_at_2023, updated_at_2022])

# Data untuk kolom applicant_category dan ticket_category
applicant_category = np.random.choice(['perorangan', 'perusahaan', 'ngo'], size=total_rows)
ticket_category = np.random.choice(['akademik', 'keuangan', 'pkm', 'perencanaan'], size=total_rows)
completion_time = np.random.randint(1, 15, size=total_rows)  # Waktu penyelesaian dalam hari
status = np.random.choice(['closed', 'open'], size=total_rows)
churn = np.random.choice(['Yes', 'No'], size=total_rows)

# Membuat DataFrame
data_example = {
    'ticket_id': range(1, total_rows + 1),
    'applicant_category': applicant_category,
    'ticket_category': ticket_category,
    'completion_time': completion_time,
    'status': status,
    'churn': churn,
    'created_at': created_at_dates,
    'updated_at': updated_at_dates
}

df = pd.DataFrame(data_example)

# Menyimpan DataFrame ke file CSV
csv_file_path = 'tiket_permohonan_with_dates_120_rows.csv'
df.to_csv(csv_file_path, index=False)

print(f"File CSV telah dibuat: {csv_file_path}")
