# line_count.py: count the lines coming in
import sys

count = 0
for line in sys.stdin:
    count += 1

print(count)  # print() also goes to stdout
