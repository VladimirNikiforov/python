import sys


def wc(filename):
    count = 0
    with open(filename) as f:
        for _ in f:
            count += 1


def process_file(filename):
    count = wc(filename)
    print(f"file: {filename} has {count} lines")


def _main():
    process_file(sys.argv[1])


if __name__ == "__main__":
    _main()
