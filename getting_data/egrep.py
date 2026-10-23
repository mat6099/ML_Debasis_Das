# egrep.py: print only the lines that match a pattern (regex)
import sys, re

regex = sys.argv[1]            # sys.argv[1] = the pattern given on the command line

for line in sys.stdin:         # read every line coming IN
    if re.search(regex, line):
        sys.stdout.write(line) # send matching lines OUT
