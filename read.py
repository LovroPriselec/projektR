#napraviti kako se može čitati da podaci dolaz sa esp ili čega već na raspberry pi
#nakon čitanja upisivati u csv, najbolje primati na stdin a slati na out koji postavit na tu csv datoteku
import sys
import time
import csv

CSV_FILE="data/temp.csv"

data = sys.stdin.readline()

with open(CSV_FILE, mode="w", newline="") as csvfile: #ne smije pisati od početka
    writer = csv.writer(csvfile)
    for line in sys.stdin:
        try:
            temp=float(line.strip())
            timestamp=int(time.time())

            writer.writerow([timestamp, temp])

            print(f"Spremljeno: {timestamp}, {temp}")
        except ValueError:
            print("Neispravan unos ", line.strip())