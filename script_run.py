import sys;

print("Hello " + ("World!" if len(sys.argv) == 1 else sys.argv[1]))
