#!/usr/bin/env python3

import scipy.io
import sys

def main():
  meta_mat = scipy.io.loadmat(sys.argv[1])
  wordnet_to_ilsvrc = []
  for row in meta_mat["synsets"]:
    ilsvrc2012_id = row[0][0][0][0]
    wordnet_id = row[0][1][0]
    if ilsvrc2012_id >= 1 and ilsvrc2012_id <= 1000:
      wordnet_to_ilsvrc.append((wordnet_id, ilsvrc2012_id))
  with open("wordnet_ids.txt", "w") as f:
    for x, _ in wordnet_to_ilsvrc:
      print("{}".format(x), file=f)
  with open("ilsvrc2012_ids.txt", "w") as f:
    for _, y in wordnet_to_ilsvrc:
      print("{}".format(y), file=f)

if __name__ == "__main__":
  main()
