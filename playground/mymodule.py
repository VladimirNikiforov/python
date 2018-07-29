def multiply(a, b):
    return a * b


"""

from mypackage.utils import multiply as mlt

if __name__ == "__main__":
    print(mlt(2, 3))
"""
# week_2
empty_list = []
empty_list = list()

none_list = [None] * 10

collections = ['list', 'tuple', 'dict', 'set']

user_data = [
    ['Elena', 4.4],
    ['Andrey', 4.2]
]

len(collections)  # constant time!

print(collections)
print(collections[0])
print(collections[-1])

collections[3] = 'frozenset'
print(collections)

range_list = list(range(10))
print(range_list)

print(range_list[1:3])
print(range_list[3:])
print(range_list[:5])
print(range_list[::2])
print(range_list[::-1])
print(range_list[5:1:-1])
print(range_list[:] is range_list)

collections = ['list', 'tuple', 'dict', 'set']
for collection in collections:
    print('Learning {}...'.format(collection))

for idx, collection in enumerate(collections):
    print('#{} {}'.format(idx, collection))

collections.append('OrderedDict')
print(collections)

collections.extend(['ponyset', 'unicorndict'])
print(collections)

collections += [None]
print(collections)

del collections[4]
print(collections)

numbers = [4, 17, 19, 9, 2, 6, 10, 13]
print(min(numbers))
print(max(numbers))
print(sum(numbers))

tag_list = ['python', 'course', 'coursera']
print(', '.join(tag_list))

####################################
# sorting
import random

numbers = []
for _ in range(10):
    numbers.append(random.randint(1, 20))

print(numbers)

print(sorted(numbers))
print(numbers)
numbers.sort()
print(numbers)
print(sorted(numbers, reverse=True))
numbers.sort(reverse=True)
print(numbers)
print(reversed(numbers))
print(list(reversed(numbers)))

""" METHODS
append
clear
copy
count
extend
index
insert
pop
remove
reverse
sort
"""

# Tuples - immutable!

empty_tuple = ()
empty_tuple = tuple()

immutables = (int, str, tuple)

# but objects inside the tuples could be mutable!
blink = ([], [])
blink[0].append(0)
print(blink)

print(hash(tuple()))

one_element_tuple = (1,)
guess_what = (1)
print(type(guess_what))
