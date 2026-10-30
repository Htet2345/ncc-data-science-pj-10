"""Recreate the fictional mall dataset; no real customer information is used."""

import csv
import random
from pathlib import Path


def generate(destination: Path):
    rng = random.Random(2026)
    observations = []
    for income_center, score_center in [(25, 20), (25, 80), (60, 50), (100, 20), (100, 80)]:
        for _ in range(40):
            income = round(min(150, max(10, rng.gauss(income_center, 6))), 2)
            score = int(round(min(100, max(1, rng.gauss(score_center, 7)))))
            observations.append((income, score))
    rng.shuffle(observations)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["customer_id", "annual_income_k", "spending_score"])
        for index, (income, score) in enumerate(observations, start=1):
            writer.writerow([f"DEMO{index:04d}", income, score])


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent.parent / "sample_data" / "customers.csv")
