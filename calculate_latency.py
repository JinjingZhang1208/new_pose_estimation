import csv

values = []

# Open the CSV file
with open('latency_log.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        values.append(float(row[1]))

average = sum(values) / len(values)

print("The average time is:", average)
