import os
import shutil
import time

source_folder = r'E:\TPCOS\registration'
destination_folder = r'C:\TPCOS\registration'
interval_minutes = 5


def move_pdf_files():
    pdf_files = [file for file in os.listdir(source_folder) if file.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        source_path = os.path.join(source_folder, pdf_file)
        destination_path = os.path.join(destination_folder, pdf_file)

        # Move the PDF file
        shutil.move(source_path, destination_path)
        print(f"Moved '{pdf_file}' to '{destination_folder}'")


while True:
    move_pdf_files()
    print(f"Waiting for {interval_minutes} minutes...")
    time.sleep(interval_minutes * 60)  # Convert minutes to seconds
