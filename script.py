import csv

input_file = "results.csv"       # your original CSV
output_file = "output.csv"     # the cleaned CSV you want to generate

columns_to_remove = {"MAE", "MSE", "MAPE"}

with open(input_file, "r", newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    # Keep only columns not in columns_to_remove
    kept_columns = [col for col in reader.fieldnames if col not in columns_to_remove]

    writer = csv.DictWriter(outfile, fieldnames=kept_columns)
    writer.writeheader()

    for row in reader:
        # Remove the unwanted columns
        filtered_row = {col: row[col] for col in kept_columns}
        writer.writerow(filtered_row)

print("Done! Cleaned file saved as:", output_file)
