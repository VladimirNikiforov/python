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

l_collections = ['list', 'tuple', 'dict', 'set']

user_data = [
    ['Elena', 4.4],
    ['Andrey', 4.2]
]

len(l_collections)  # constant time!

print(l_collections)
print(l_collections[0])
print(l_collections[-1])

l_collections[3] = 'frozenset'
print(l_collections)

range_list = list(range(10))
print(range_list)

print(range_list[1:3])
print(range_list[3:])
print(range_list[:5])
print(range_list[::2])
print(range_list[::-1])
print(range_list[5:1:-1])
print(range_list[:] is range_list)

l_collections = ['list', 'tuple', 'dict', 'set']
for collection in l_collections:
    print('Learning {}...'.format(collection))

for idx, collection in enumerate(l_collections):
    print('#{} {}'.format(idx, collection))

l_collections.append('OrderedDict')
print(l_collections)

l_collections.extend(['ponyset', 'unicorndict'])
print(l_collections)

l_collections += [None]
print(l_collections)

del l_collections[4]
print(l_collections)

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

# Dictionaries

empty_dict = {}
empty_dict = dict()

collections_map = {
    'mutable': ['list', 'set', 'dict'],
    'immutable': ['tuple', 'frozenset']
}

print(collections_map['immutable'])
print(collections_map.get('irresistible', 'not found'))

print('mutable' in collections_map)

beatles_map = {
    'Paul': 'Bass',
    'John': 'Guitar',
    'George': 'Guitar',
}

print(beatles_map)
beatles_map['Ringo'] = 'Drums'
print(beatles_map)

del beatles_map['John']
print(beatles_map)

beatles_map.update({
    'John': 'Guitar'
})
print(beatles_map)

print(beatles_map.pop('Ringo'))
print(beatles_map)

unknown_dict = {}
print(unknown_dict.setdefault('key', 'default'))
print(unknown_dict)
print(unknown_dict.setdefault('key', 'new_default'))

for key in collections_map:
    print(key)

for key, value in collections_map.items():
    print('{} - {}'.format(key, value))

for value in collections_map.values():
    print(value)

from collections import OrderedDict

ordered = OrderedDict()

for number in range(10):
    ordered[number] = str(number)

for key in ordered:
    print(key)

#######################
# Sets

empty_set = set()
number_set = {1, 2, 3, 3, 4, 5}
print(number_set)
print(2 in number_set)

odd_set = set()
even_set = set()
for number in range(10):
    if number % 2:
        odd_set.add(number)
    else:
        even_set.add(number)

print(odd_set)
print(even_set)

# Union of sets
union_set = odd_set | even_set
union_set = odd_set.union(even_set)
print(union_set)

# Intersection of sets
intersection_set = odd_set & even_set
intersection_set = odd_set.intersection(even_set)
print(intersection_set)

# difference between sets
difference_set = odd_set - even_set
difference_set = odd_set.difference(even_set)
print(difference_set)

# symmetric difference between sets
symmetric_difference_set = odd_set ^ even_set
symmetric_difference_set = odd_set.symmetric_difference(even_set)
print(symmetric_difference_set)

even_set.remove(2)
print(even_set)

print(even_set.pop())
print(even_set)

frozen = frozenset(['A', 'B', 'C'])  # immutable!
# frozen.add('O') # Error!
