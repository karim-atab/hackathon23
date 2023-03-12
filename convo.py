import csv 
with open('convo.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("|") for line in stripped if line)
    with open('convo.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('ConvoID','CallerID','Script'))
        writer.writerows(lines)