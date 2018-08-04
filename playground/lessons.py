def week_1():
    def multiply(a, b):
        return a * b


"""

from mypackage.utils import multiply as mlt

if __name__ == "__main__":
    print(mlt(2, 3))
"""


def week_2():
    def lists():

        empty_list = []
        empty_list = list()

        none_list = [None] * 10

        l_collections = ['list', 'tuple', 'dict', 'set']

        print('list' in l_collections)  # searching element in list need linear time!

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
    def sorting():
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

    def tuples():
        # Tuples - immutable!
        # неизменяемые, могут содержать эл-ты разных типов, поиск за линейное время
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

    def dictionaries():

        empty_dict = {}
        empty_dict = dict()

        collections_map = {
            'mutable': ['list', 'set', 'dict'],
            'immutable': ['tuple', 'frozenset']
        }

        print(collections_map['immutable'])
        print(collections_map.get('irresistible', 'not found'))

        print('mutable' in collections_map)  # searching key in dict need constant time!

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
    def sets():
        # изменяемые, поиск элемента за константное время
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

        # Usefull links
        # https://docs.python.org/3/library/stdtypes.html
        # https://docs.python.org/3/tutorial/datastructures.html
        # https://en.wikipedia.org/wiki/Hash_table

    def functions():
        from datetime import datetime

        def get_seconds():
            """Return current seconds"""
            return datetime.now().second

        # print(get_seconds())

        def split_tags(tag_string):
            tag_list = []
            for tag in tag_string.split(','):
                tag_list.append(tag.strip())

            return tag_list

        # print(split_tags('python, coursera, mooc'))

        # Types annotation!
        def add(x: int, y: int) -> int:
            return x + y

        # print(add(10, 11))
        # print(add('still ', 'works'))

        def extender(source_list, extend_list):
            source_list.extend(extend_list)

        # values = [1, 2, 3]
        # extender(values, [4, 5, 6])
        # print(values)

        def replacer(source_tuple, replace_with):
            source_tuple = replace_with

        # user_info = ('Guido', '31/01')
        # replacer(user_info, ('Larry', '27/09'))
        # print(user_info)

        # Named args
        def say(greeting, name):
            print('{} {}'.format(greeting, name))

        # say('Hello', 'Kitty')
        # say(name='Kitty', greeting='Hello')

        # Args by default
        def greeting(name='it\'s me...'):
            print('Hello, {}'.format((name)))

        # greeting()

        def append_one(iterable=[]):
            iterable.append(1)
            return iterable

        # print(append_one([1]))
        # print(append_one()) # [1]
        # print(append_one()) # [1, 1]

        # def function(iterable=None):
        #    if iterable is None:
        #        iterable = []
        # def function(iterable=None):
        #    iterable = iterable or []

        # STARS
        def printer(*args):
            print(type(args))

            for argument in args:
                print(argument)

        # printer(1, 2, 3, 4, 5)
        # name_list = ['John', 'Bill', 'Amy']
        # printer(*name_list)

        def printer(**kwargs):
            print(type(kwargs))

            for key, value in kwargs.items():
                print('{}: {}'.format(key, value))

        # printer(a=10, b=11) # a: 10 / b: 11

        payload = {
            'user_id': 117,
            'feedback': {
                'subject': 'Registration fields',
                'message': 'There is no country for old men'
            }
        }
        # printer(**payload)
        # <class 'dict'>
        # user_id: 117
        # feedback: {'subject': 'Registration fields', 'message': 'There is no country for old men'}

    def files():
        f = open('filename')
        text_model = ['r', 'w', 'a', 'r+']
        binary_modes = ['br', 'bw', 'ba', 'br+']
        f.write('The world is changed.\nI taste it in the water.\n')  # 47
        f.close()
        f.open('filename', 'r+')
        f.read()
        f.tell()  # 47
        f.read()  # ..
        f.seek(0)
        f.tell()  # 0
        print(f.read())  # The world is changed.\n
        f.close()

        f = open('filename', 'r+')
        f.readline()  # 'The world is changed.\n'
        f.close()
        f = open('filename', 'r+')
        f.readlines()  # ['The world is changed.\n','I taste it in the water.\n']
        f.close()

        with open('filename') as f:
            print(f.read())

    def functional_programming():
        def caller(func, params):
            return func(*params)

        def printer(name, origin):
            print('I\'m {} of {}!'.format(name, origin))

        # caller(printer, ['Moana', 'Motunui'])

        def get_multiplier():
            def inner(a, b):
                return a * b

            return inner

        # multiplier = get_multiplier()
        # print(multiplier(10, 11))
        # print(multiplier.__name__)

        def get_multiplier(number):
            def inner(a):
                return a * number

            return inner

        # multiplier_by_2 = get_multiplier(2)
        # print(multiplier_by_2(10))

        #######################################################
        # map, filter, lambda
        def squarify(a):
            return a ** 2

        # print(list(map(squarify, range(5)))) # [0, 1, 4, 9, 16]
        # OLD STYLE:
        squared_list = []
        for number in range(5):
            squared_list.append(squarify(number))

        # print(squared_list) # [0, 1, 4, 9, 16]

        def is_positive(a):
            return a > 0

        # print(list(filter(is_positive, range(-2, 3))))
        # OLD STYLE:
        positive_list = []
        for number in range(-2, 3):
            if is_positive(number):
                positive_list.append(number)

        # print(positive_list)
        # LAMBDA-function
        # print(list(map(lambda x: x ** 2, range(5))))
        # print(list(filter(lambda  x: x > 0, range(-2, 3))))
        def stringify_list(num_list):
            return list(map(str, [1, 2, 3]))

        # print(stringify_list(range(10)))

        # FUNCTOOLS
        def multiply(a, b):
            return a * b

        # print(reduce(multiply, [1, 2, 3, 4, 5]))
        # print(reduce(lambda x, y: x * y, range(1, 6)))
        from functools import partial
        def greeter(person, greeting):
            return '{}, {}!'.format(greeting, person)

        hier = partial(greeter, greeting='Hi')
        helloer = partial(greeter, greeting='Hello')
        # print(hier('brother'))
        # print(helloer('sir'))

        # Create List
        # print([number ** 2 for number in range(10)])
        # print([num for num in range(10) if num % 2 == 0])
        # Create Dict
        # print({number: number ** 2 for number in range(5)})

        # print({num % 10 for num in range(100)})
        num_list = range(7)
        squared_list = [x ** 2 for x in num_list]
        print(list(zip(num_list, squared_list)))
        # [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25), (6, 36)]

    def decorators():
        """
        def decorator(func):
            return func
        @decorator
        def decorated():
            print('Hello!')
        decorated = decorator(decorated)
        """
        """
        def decorator(func):
            def new_func():
                pass
            return new_func

        @decorator
        def decorated():
            print('Hello!')
        decorated()
        print(decorated.__name__)
        """
        """
        # Need decorator that writes result of decorated function to log
        
        import functools
        def logger(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                result = func(*args, **kwargs)
                with open('log.txt','w') as f:
                    f.write(str(result))

                return result
            return wrapped

        @logger
        def summator(num_list):
            return sum(num_list)
        print('Summator: {}'.format(summator([1, 2, 3, 4, 5])))
        with open('log.txt', 'r') as f:
            print('log.txt: {}'.format(f.read()))
        print(summator.__name__)
        """
        """
        # Need decorator with param, which writes log to specified file
        
        def logger(filename):
            def decorator(func):
                def wrapped(*args, **kwargs):
                    result = func(*args, **kwargs)
                    with open(filename, 'w') as f:
                        f.write(str(result))
                    return result
                return wrapped
            return decorator

        @logger('new_log.txt')
        def summator(num_list):
            return sum(num_list)

        print('Summator: {}'.format(summator([1, 2, 3, 4, 5, 6])))
        with open('new_log.txt', 'r') as f:
            print('new_log.txt: {}'.format(f.read()))
        """

        """
        If we use many decorators?
        
        def first_decorator(func):
            def wrapped():
                print('Inside first_decorator product')
                return func()
            return wrapped
        def second_decorator(func):
            def wrapped():
                print('Inside second_decorator product')
                return func()
            return wrapped

        @first_decorator
        @second_decorator
        def decorated():
            print('Finally called...')
        # decorated = first_decorator(second_decorator(decorated))
        decorated()
        #Inside first_decorator product
        #Inside second_decorator product
        #Finally called...
        """

        def bold(func):
            def wrapped():
                return "<b>" + func() + "</b>"

            return wrapped

        def italic(func):
            def wrapped():
                return "<i>" + func() + "</i>"

            return wrapped

        @bold
        @italic
        def hello():
            return "hello world"

        # hello = bold(italic(hello))
        print(hello())
        # <b><i>hello world</i></b>

    def generators():
        def even_range(start, end):
            current = start
            while current < end:
                yield current
                current += 2

        # for number in even_range(0, 10):
        #    print(number)
        ranger = even_range(0, 4)

        # print(next(ranger))
        # print(next(ranger))

        def list_generator(list_obj):
            for item in list_obj:
                yield item
                print('After yielding {}'.format(item))

        generator = list_generator([1, 2])

        # print(next(generator))
        # print(next(generator))

        def fibonacci(number):
            a = b = 1
            for _ in range(number):
                yield a
                a, b = b, a + b

        # for num in fibonacci(10):
        #    print(num)

        def accumulator():
            total = 0
            while True:
                value = yield total
                print('Got: {}'.format(value))

                if not value: break
                total += value

        generator = accumulator()
        print(next(generator))

        print('Accumulated: {}'.format(generator.send(1)))
        print('Accumulated: {}'.format(generator.send(5)))
        print('Accumulated: {}'.format(generator.send(4)))

        # Туториал по функциям из документации
        # https://docs.python.org/3/tutorial/controlflow.html#defining-functions

        # Работа с файлами
        # https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files

        # Встроенные функции
        # https://docs.python.org/3/library/functions.html

        # Сортировка
        # https://docs.python.org/3/howto/sorting.html

        # Функциональное программирование
        # https://docs.python.org/3/howto/functional.html

        # Модуль functools
        # https://docs.python.org/3/library/functools.html

        # Декораторы
        # http://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html

    generators()


def week_3():
    # num = 13
    # print(isinstance(num, int))
    class Human:
        pass

    class Robot:
        """Instead pass we use some docs"""

    # print(Robot)
    # print(dir(Robot))
    class Planet:
        pass

    planet = Planet()
    # print(planet)
    solar_system = []
    for i in range(8):
        planet = Planet()
        solar_system.append(planet)
    # print(solar_system)
    solar_system = {}
    for i in range(8):
        planet = Planet()
        solar_system[planet] = True

    # print(solar_system)

    class Planet:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return self.name

        def __repr__(self):
            return f"Planet {self.name}"

    earth = Planet("Earth")
    # print(earth.name) #Earth
    # print(earth) # with __str__ => Earth

    solar_system = []
    planet_names = [
        "Mercury", "Venus", "Earth", "Mars",
        "Jupyter", "Saturn", "Uranus", "Neptune"
    ]

    for name in planet_names:
        planet = Planet(name)
        solar_system.append(planet)
    # print(solar_system) # [<__main__.week_3.<locals>.Planet object at 0x034FDA70>, <__main__.week_3.<locals>.Planet object at 0x034FDB90>, <__main__.week_3.<locals>.Planet object at 0x034FDA50>, <__main__.week_3.<locals>.Planet object at 0x034FD9D0>, <__main__.week_3.<locals>.Planet object at 0x034FDAB0>, <__main__.week_3.<locals>.Planet object at 0x034FDC10>, <__main__.week_3.<locals>.Planet object at 0x034FDC30>, <__main__.week_3.<locals>.Planet object at 0x034FDC50>]
    # with __repr__ => [Planet Mercury, Planet Venus, Planet Earth, Planet Mars, Planet Jupyter, Planet Saturn, Planet Uranus, Planet Neptune]

    mars = Planet("Mars")
    # print(mars) # Planet Mars
    # print(mars.name) # 'Mars'

    mars.name = "Second Earth?"
    # print(mars.name) # 'Second Earth?'
    # del mars.name => delete attribute!

    ##### Attributes for class
    class Planet:
        """Some description for class Planet"""
        count = 0

        def __init__(self, name, population=None):
            self.name = name
            self.population = population or []
            Planet.count += 1

    earth = Planet("Earth")
    mars = Planet("Mars")

    # print(Planet.count) #2
    # print(mars.count) #2

    # destructors for class
    class Human:

        def __del__(self):
            print("Goodbye!")

    # human = Human()
    # del human # Goodbye!
    planet = Planet("Earth")

    # print(planet.__dict__) # {'name': 'Earth', 'population': []}
    # planet.mass = 5.97e24
    # print(planet.__dict__)  # {'name': 'Earth', 'population': [], 'mass': 5.97e+24}
    # print(Planet.__dict__) # {'__module__': '__main__', '__doc__': 'Some description for class Planet', 'count': 3, '__init__': <function week_3.<locals>.Planet.__init__ at 0x015FCA50>, '__dict__': <attribute '__dict__' of 'Planet' objects>, '__weakref__': <attribute '__weakref__' of 'Planet' objects>}
    # print(Planet.__doc__) # Some description for class Planet
    # print(dir(planet)) # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'count', 'name', 'population']
    # print(planet.__class__) # <class '__main__.week_3.<locals>.Planet'>

    # constructor class unit
    class Planet:

        def __new__(cls, *args, **kwargs):
            print("__new__ called")
            obj = super().__new__(cls)
            return obj

        def __init__(self, name):
            print("__init__ called")
            self.name = name

    planet = Planet("Earth")
    # == planet = Planet.__new__(Planet, "Earth")
    # if isinstance(planet, Planet):
    #     Planet.__init__(planet, "Earth")


week_3()
