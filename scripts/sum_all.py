import os
from glob import glob


def main():
    fname = "summary"
    if os.path.exists(fname):
        os.remove(fname)
    pbs = sorted(glob("*.pb"))
    for pb in pbs:
        os.system("summarize_graph --in_graph=%s >> %s" % (pb, fname))


if __name__ == "__main__":
    main()
