import csv


data = []
for i in range(296, 1410):
    for j in range(518, 987):
        row = []
        row.append(i)
        row.append(j)
        data.append(row)
f = open('data/q2_traverse.csv', 'w')
with f:
    writer = csv.writer(f)
    writer.writerows(data)


