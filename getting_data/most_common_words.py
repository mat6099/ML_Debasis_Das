# most_common_words.py: show the N most common words
import sys
from collections import Counter

try:
    num_words = int(sys.argv[1])
except (IndexError, ValueError):
    print("usage: most_common_words.py num_words")
    sys.exit(1)                # non-zero exit code = error

counter = Counter(word.lower()
                  for line in sys.stdin
                  for word in line.strip().split()
                  if word)

for word, count in counter.most_common(num_words):
    sys.stdout.write(f"{count}\t{word}\n")
