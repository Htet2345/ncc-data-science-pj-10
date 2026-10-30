# Synthetic mall customer data

`customers.csv` contains 200 fictional observations, generated with Python's
standard library and seed 2026. No actual customer records or identities are used.

Features are annual income in thousands of synthetic USD and an invented
integer spending score from 1 to 100. The score is not a dollar amount or a
percentage. IDs beginning with DEMO are audit labels only, not model features.

The generator places 40 observations around each of five illustrative centers:
(25,20), (25,80), (60,50), (100,20), (100,80), then shuffles the rows. Gaussian
standard deviations are 6 income units and 7 score units. Generated incomes are
clipped to 10-150 and rounded to two decimals; scores are clipped to 1-100 and
rounded to whole numbers. The generating group labels are not stored or used
to fit the model or select K.

Run `python -m scripts.generate_data` from the project folder to recreate the CSV.
The deliberately separated groups make this suitable for learning K-means,
not for demonstrating real-world segmentation quality. The data has not been
confirmed as the teacher's dataset.
