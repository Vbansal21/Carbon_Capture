import csv
import random
import sys

def generate_csv(filename, n_rows):
    # Define column names
    column_names = [f'input_{i+1}' for i in range(10)] + ['output']
    
    # Generate data
    data = []
    for _ in range(n_rows):
        row = [random.random() for _ in range(10)]  # Generate random input data
        output = sum(row)  # Example output calculation (sum of input data)
        row.append(output)
        data.append(row)
    
    # Write data to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)  # Write column names
        writer.writerows(data)  # Write data rows

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_csv.py <filename> <n_rows>")
        #sys.exit(1)

    filename = "data.csv"#sys.argv[1]
    n_rows = 573*5#int(sys.argv[2])

    generate_csv(filename, n_rows)
    print(f"CSV file '{filename}' with {n_rows} rows generated successfully.")
    