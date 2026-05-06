import numpy as np
from config import *
from utils import ensure_dir_for_file
from build_index import build_index
from search_index import search_all


def main():
    print("CPUs:", N_CPUS)

    build_index()

    neighbors, scores, ids = search_all()

    ensure_dir_for_file(NEIGHBORS_PATH)
    np.save(NEIGHBORS_PATH, neighbors)
    np.save(SCORES_PATH, scores)

    print("Example:")
    print("Query id:", ids[0])
    print("Neighbor ids:", ids[neighbors[0][:10]])
    print("Scores:", scores[0][:10])

    print("Saved:")
    print(" ", NEIGHBORS_PATH)
    print(" ", SCORES_PATH)


if __name__ == "__main__":
    main()
