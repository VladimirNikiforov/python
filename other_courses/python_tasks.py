# Read a full line of input from stdin and save it to our dynamically typed variable, input_string.
input_string = input()

# Print a string literal saying "Hello, World." to stdout.
print('Hello, World.')

# TODO: Write a line of code here that prints the contents of input_string to stdout.
print(input_string)

i = 4
d = 4.0
s = 'HackerRank '
# Declare second integer, double, and String variables.
ii = 5
dd = 3.0
ss = 'is the Best!'
# Read and save an integer, double, and String to your variables.
ii = int(input())
dd = float(input())
ss = str(input())
# Print the sum of both integer variables on a new line.
print(i + ii)
# Print the sum of the double variables on a new line.
print(d + dd)
# Concatenate and print the String variables on a new line
# The 's' variable above should be printed first.
print(s + ss)

# !/bin/python3

if __name__ == "__main__":
    meal_cost = float(input().strip())
    tip_percent = int(input().strip())
    tax_percent = int(input().strip())
    total = meal_cost * (100. + tip_percent + tax_percent) / 100.
    print("The total meal cost is {:d} dollars.".format(round(total)))

# !/bin/python3


N = int(input().strip())
if N % 2 == 1:
    print('Weird')
else:
    if (N >= 2) and (N <= 5):
        print('Not Weird')
    else:
        if (N >= 6) and (N <= 20):
            print('Weird')
        else:
            print('Not Weird')


###################################
class Person:
    def __init__(self, initialAge):
        # Add some more code to run some checks on initialAge
        if initialAge < 0:
            self.age = 0
            print('Age is not valid, setting age to 0.')
        else:
            self.age = initialAge

    def amIOld(self):
        # Do some computations in here and print out the correct statement to the console
        if self.age < 13:
            print('You are young.')
        else:
            if (self.age >= 13) and (self.age < 18):
                print('You are a teenager.')
            else:
                print('You are old.')

    def yearPasses(self):
        # Increment the age of the person in here      
        self.age += 1


t = int(input())
for i in range(0, t):
    age = int(input())
    p = Person(age)
    p.amIOld()
    for j in range(0, 3):
        p.yearPasses()
    p.amIOld()
    print("")

##############################################
# !/bin/python3


n = int(input().strip())
for i in range(1, 10 + 1):
    print('%d x %d = %d' % (n, i, n * i))

##############################################
# !/bin/python3

t = int(input().strip())
for i in range(0, t):
    s = input()
    s1 = ''
    s2 = ''
    for j in range(len(s)):
        if j % 2 == 0:
            s1 += s[j]
        else:
            s2 += s[j]
    print(s1, s2)
################################################
# !/bin/python3


n = int(input().strip())
arr = [int(arr_temp) for arr_temp in input().strip().split(' ')]
print(*arr[::-1])
################################################
# !/bin/python3

import sys

n = int(input().strip())
d = {}
for i in range(n):
    dd = input().split(' ')
    d[dd[0]] = dd[1]
for k in sys.stdin:
    if k.strip() in d:
        print('%s=%s' % (k.strip(), d[k.strip()]))
    else:
        print('Not found')


################################################
# !/bin/python3

def factorial(n):
    # Complete this function
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


if __name__ == "__main__":
    n = int(input().strip())
    result = factorial(n)
    print(result)

################################################
# !/bin/python3

n = int(input().strip())
nb = "{0:b}".format(n)
m = 0
mmax = 0
for i in range(len(nb)):
    if nb[i] == '1':
        m += 1
        mmax = m if mmax < m else mmax
    else:
        m = 0
print(mmax)
################################################
# !/bin/python3


arr = []
for arr_i in range(6):
    arr_t = [int(arr_temp) for arr_temp in input().strip().split(' ')]
    arr.append(arr_t)
maxsum = -999
sum = 0
for i in range(1, (len(arr) - 1)):
    for j in range(1, (len(arr[i]) - 1)):
        sum = arr[i - 1][j - 1] + arr[i - 1][j] + arr[i - 1][j + 1] + \
              arr[i][j] + \
              arr[i + 1][j - 1] + arr[i + 1][j] + arr[i + 1][j + 1]
        if sum > maxsum:
            maxsum = sum
print(maxsum)


################################################
class Person:
    def __init__(self, firstName, lastName, idNumber):
        self.firstName = firstName
        self.lastName = lastName
        self.idNumber = idNumber

    def printPerson(self):
        print("Name:", self.lastName + ",", self.firstName)
        print("ID:", self.idNumber)


class Student(Person):
    #   Class Constructor
    #   
    #   Parameters:
    #   firstName - A string denoting the Person's first name.
    #   lastName - A string denoting the Person's last name.
    #   id - An integer denoting the Person's ID number.
    #   scores - An array of integers denoting the Person's test scores.
    #
    # Write your constructor here
    def __init__(self, firstName, lastName, idNumber, scores):
        self.firstName = firstName
        self.lastName = lastName
        self.idNumber = idNumber
        self.scores = scores

    #   Function Name: calculate
    #   Return: A character denoting the grade.
    #
    # Write your function here
    def calculate(self):
        avg_ = sum(self.scores) / len(self.scores)
        if avg_ < 40:
            return 'T'
        elif avg_ < 55:
            return 'D'
        elif avg_ < 70:
            return 'P'
        elif avg_ < 80:
            return 'A'
        elif avg_ < 90:
            return 'E'
        else:
            return 'O'


line = input().split()
firstName = line[0]
lastName = line[1]
idNum = line[2]
numScores = int(input())  # not needed for Python
scores = list(map(int, input().split()))
s = Student(firstName, lastName, idNum, scores)
s.printPerson()
print("Grade:", s.calculate())
################################################
################################################
################################################
# Abstract classes
from abc import ABCMeta, abstractmethod


class Book(object, metaclass=ABCMeta):
    def __init__(self, title, author):
        self.title = title
        self.author = author

    @abstractmethod
    def display(): pass


# Write MyBook class
class MyBook(Book):
    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def display(self):
        print('Title: %s' % (self.title))
        print('Author: %s' % (self.author))
        print('Price: %d' % (self.price))


title = input()
author = input()
price = int(input())
new_novel = MyBook(title, author, price)
new_novel.display()


################################################
################################################
################################################
class Difference:
    def __init__(self, a):
        self.__elements = a

    # Add your code here
    def computeDifference(self):
        self.maximumDifference = sorted(a)[-1] - sorted(a)[0]


# End of Difference class

_ = input()
a = [int(e) for e in input().split(' ')]

d = Difference(a)
d.computeDifference()

print(d.maximumDifference)


################################################
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Solution:
    def display(self, head):
        current = head
        while current:
            print(current.data, end=' ')
            current = current.next

    def insert(self, head, data):
        # Complete this method
        if head is None:
            head = Node(data)
        else:
            current = head
            while current.next:
                current = current.next
            current.next = Node(data)
        return (head)


mylist = Solution()
T = int(input())
head = None
for i in range(T):
    data = int(input())
    head = mylist.insert(head, data)
mylist.display(head);
################################################
################################################
# !/bin/python3


S = input().strip()
try:
    print(int(S))
except Exception:
    print("Bad String")


################################################
################################################
# !/bin/python3

def solve(grades):
    # Complete this function
    return map(lambda x: x if x < 38 else (x if x % 5 < 3 else 5 * ((x // 5) + 1)), grades)


n = int(input().strip())
grades = []
grades_i = 0
for grades_i in range(n):
    grades_t = int(input().strip())
    grades.append(grades_t)
result = solve(grades)
print("\n".join(map(str, result)))


################################################
class Calculator:
    def power(self, n, p):
        if n < 0 or p < 0:
            return ('n and p should be non-negative')
        else:
            return (n ** p)


myCalculator = Calculator()
T = int(input())
for i in range(T):
    n, p = map(int, input().split())
    try:
        ans = myCalculator.power(n, p)
        print(ans)
    except Exception as e:
        print(e)
    ################################################


# palindrom
class Solution:
    # Write your code here
    stack = []
    queue = []

    def pushCharacter(self, x):
        self.stack.append(x)

    def enqueueCharacter(self, x):
        self.queue.append(x)

    def popCharacter(self):
        return (self.stack[-1])

    def dequeueCharacter(self):
        return (self.queue[0])


# read the string s
s = input()
# Create the Solution class object
obj = Solution()

l = len(s)
# push/enqueue all the characters of string s to stack
for i in range(l):
    obj.pushCharacter(s[i])
    obj.enqueueCharacter(s[i])

isPalindrome = True
'''
pop the top character from stack
dequeue the first character from queue
compare both the characters
'''
for i in range(l // 2):
    if obj.popCharacter() != obj.dequeueCharacter():
        isPalindrome = False
        break
# finally print whether string s is palindrome or not.
if isPalindrome:
    print("The word, " + s + ", is a palindrome.")
else:
    print("The word, " + s + ", is not a palindrome.")
##########################################################
if __name__ == '__main__':
    print
    "Hello, World!"
###########################
if __name__ == '__main__':
    n = int(raw_input())
if n % 2 == 1:
    print("Weird")
else:
    if n >= 2 and n <= 5:
        print("Not Weird")
    if n >= 6 and n <= 20:
        print("Weird")
    if n > 20:
        print("Not Weird")

###########################
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a + b)
    print(a - b)
    print(a * b)

###########################
from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a // b)
    print(a / b)

###########################
if __name__ == '__main__':
    n = int(raw_input())
    for i in range(n):
        print(i ** 2)


###########################
def is_leap(year):
    leap = False

    # Write your logic here
    if year % 4 == 0:
        if year % 400 == 0:
            leap = True
        else:
            if year % 100 == 0:
                leap = False
            else:
                leap = True
    else:
        leap = False
    return leap


year = int(input())
print(is_leap(year))
###########################
# N=3 => 123
if __name__ == '__main__':
    n = int(input())
    print(*range(1, n + 1), sep='')

###########################
n, m = map(int, input().split())
arr = input().split(' ')
a = set(input().split(' '))
b = set(input().split(' '))
# print(len(arr.intersection(a))-len(arr.intersection(b)))
# print(len(list(filter(lambda x: x in arr, a)))-len(list(filter(lambda x: x in arr, b))))
print(sum(list(1 if x in a else -1 if x in b else 0 for x in arr)))

###########################
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s = set()
for i in range(n):
    s.add(input())
print(len(s))

###########################
n = int(input())
s = set(map(int, input().split()))
cmd_count = int(input())
for i in range(cmd_count):
    raw = input()
    if raw == 'pop':
        cmd = raw
    else:
        cmd, num = raw.split(' ')
    if cmd == 'pop':
        s.pop()
    elif cmd == 'remove':
        s.remove(int(num))
    else:
        s.discard(int(num))
print(sum(s))

###########################
n = int(input())
a = [int(i) for i in input().split()]
m = int(input())
b = [int(i) for i in input().split()]
print(len(set(a).union(set(b))))

###########################
n = int(input())
a = [int(i) for i in input().split()]
m = int(input())
b = [int(i) for i in input().split()]
print(len(set(a).intersection(set(b))))

###########################
n = int(input())
a = [int(i) for i in input().split()]
m = int(input())
b = [int(i) for i in input().split()]
print(len(set(a).difference(set(b))))

###########################

n = int(input())
a = [int(i) for i in input().split()]
m = int(input())
b = [int(i) for i in input().split()]
print(len(set(a).symmetric_difference(set(b))))
###########################
n = int(input())
arr = set([int(x) for x in input().split()])
m = int(input())
for i in range(m):
    cmd, y = input().split()
    x = set([int(t) for t in input().split()])
    if cmd == 'intersection_update':
        arr.intersection_update(x)
    elif cmd == 'update':
        arr.update(x)
    elif cmd == 'symmetric_difference_update':
        arr.symmetric_difference_update(x)
    else:
        arr.difference_update(x)
print(sum(arr))

###########################
# k = int(input())
# ar = [int(x) for x in input().split()]
# print(*set(ar[:(len(ar)//2)]).symmetric_difference(set(ar[(len(ar)//2)+1:])))
k, ar = int(input()), [int(x) for x in input().split()]
# k,ar = int(input()),list(map(int, input().split()))
arr = set(ar)
print(((sum(arr) * k) - (sum(ar))) // (k - 1))

###########################

for i in range(int(input())):  # More than 4 lines will result in 0 score. Blank lines won't be counted.
    a = int(input());
    A = set(input().split())
    b = int(input());
    B = set(input().split())
    if len(A.intersection(B)) == len(A):
        print('True')
    else:
        print('False')
###########################
# https://www.hackerrank.com/challenges/py-check-strict-superset/problem
A = set(input().split())
flag = 1
for i in range(int(input())):
    B = set(input().split())
    if len(A.intersection(B)) < len(B):
        flag = 0
        exit
print(flag == 1)

###########################
# !/bin/python3

from datetime import datetime


def time_delta(t1, t2):
    # Complete this function
    t1_date = datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
    t2_date = datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
    return (abs((t1_date - t2_date).days * 24 * 3600 + (t1_date - t2_date).seconds))


if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        t1 = input().strip()
        t2 = input().strip()
        delta = time_delta(t1, t2)
        print(delta)

###########################
# Replace all ______ with rjust, ljust or center.

thickness = int(input())  # This must be an odd number
c = 'H'

# Top Cone
for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

# Top Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

# Middle Belt
for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))

# Bottom Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

# Bottom Cone
for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(
        thickness * 6))

###########################
import textwrap


def wrap(string, max_width):
    return textwrap.fill(string, max_width)


if __name__ == '__main__':
    string, max_width = raw_input(), int(raw_input())
    result = wrap(string, max_width)
    print
    result

###########################
N, M = map(int, raw_input().split())  # More than 6 lines of code will result in 0 score. Blank lines are not counted.
for i in xrange(1, N, 2):
    print(('.|.') * i).center(M, '-')
print
'WELCOME'.center(M, '-')
for i in xrange(N - 2, -1, -2):
    print(('.|.') * i).center(M, '-')


###########################
def print_formatted(number):
    l = len("{0:b}".format(number))
    for i in range(1, number + 1):
        print
        '{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}'.format(i, width=l)


if __name__ == '__main__':
    n = int(raw_input())
    print_formatted(n)


###########################
def print_rangoli(size):
    # chars = 'abcdefghijklmnopqrstuvwxwz'
    import string
    chars = string.ascii_lowercase
    h = []
    for i in range(size):
        s = '-'.join(chars[i:size])
        h.append(((s[::-1] + s[1:]).center(2 * 2 * (size - 1) + 1, '-')))
    print
    '\n'.join(h[::-1] + h[1:])


if __name__ == '__main__':
    n = int(raw_input())
    print_rangoli(n)


###########################
def capitalize(string):
    return ' '.join([w.capitalize() for w in string.split(' ')])


###########################
def minion_game(string):
    s = 0
    k = 0
    for i in range(len(string)):
        if string[i] in 'AEIOU':
            k += len(string) - i
        else:
            s += len(string) - i
    if s > k:
        print
        'Stuart', s
    else:
        if s < k:
            print
            'Kevin', k
        else:
            print
            'Draw'
    # alternate print with implied ifelse!
    # print(['Stuart '+str(s),['Kevin '+str(k),'Draw'][k==s]][k>=s])


if __name__ == '__main__':
    s = raw_input()
    minion_game(s)


###########################
###########################
def merge_the_tools(string, k):
    s = [string[(i * k):((i + 1) * k)] for i in range(len(string) // k)]
    print('\n'.join([''.join([x for i, x in enumerate(sx) if sx.index(x) == i]) for sx in s]))


if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)
# alternative
# S, N = input(), int(input())
# for part in zip(*[iter(S)] * N):
#    d = dict()
#    print(''.join([ d.setdefault(c, c) for c in part if c not in d ]))

###########################
###########################

from itertools import combinations

s, k = input().split()
print('\n'.join(sorted([''.join(sorted(x)) for i in range(1, int(k) + 1) for x in list(combinations(s, i))],
                       key=lambda x: (len(''.join(x)), ''.join(x)))))
# alternate
# print(*[''.join(j) for i in range(1,int(b)+1) for j in combinations(sorted(a),i)],sep='\n')
###########################
###########################
from itertools import combinations_with_replacement

s, k = input().split()
print(*[''.join(x) for x in list(combinations_with_replacement(sorted(s), int(k)))], sep='\n')

###########################
###########################
from itertools import groupby

s = input()
print(*[(len(list(g)), int(k)) for k, g in groupby(s)], sep=' ')
# alternative
# from itertools import groupby
# print(*[(len(list(g)),int(k)) for k, g in groupby(input())])

###########################
###########################
from itertools import combinations

n = int(input())
s = input().split()
k = int(input())
l = [1 if 'a' in x else 0 for x in list(combinations(sorted(s), int(k)))]
print(round(sum(l) / len(l), 3))

# alternate
# from itertools import combinations
# _,s,n = input(),input().split(),int(input())
# t = list(combinations(s,n))
# f = [i for i in t if 'a' in i]
# print(len(f)/len(t))
###########################
###########################
from itertools import product

k, m = map(int, input().split())
n = []
for i in range(k):
    n.append([(int(i) ** 2) for i in input().split()[1:]])
print(max([sum(x) % m for x in product(*n)]))

# alternate
# from itertools import product
# K,M = map(int,input().split())
# N = (list(map(int, input().split()))[1:] for _ in range(K))
# results = map(lambda x: sum(i**2 for i in x)%M, product(*N))
# print(max(results))

###########################
###########################
a = int(input())
b = int(input())
m = int(input())
print(*[a ** b, pow(a, b, m)], sep='\n')

###########################
###########################
# 1
# 22
# 333
for i in range(1, int(input())):
    print(i * ((10 ** i) // 9))

###########################
###########################
import numpy

my_array = numpy.array(list(map(int, input().split())))
print(numpy.reshape(my_array, (3, 3)))

###########################
###########################
import numpy as np

n, _ = map(int, input().split())
a = np.array([list(map(int, input().split())) for _ in range(n)])
print(np.transpose(a))
print(a.flatten())

###########################
###########################
import numpy as np

n, m, p = map(int, input().split())
a = np.array([list(map(int, input().split())) for _ in range(n)])
b = np.array([list(map(int, input().split())) for _ in range(m)])

print(np.concatenate((a, b), axis=0))

###########################
###########################
import numpy as np

a = [int(i) for i in input().split()]
print(np.zeros(a, dtype=np.int))
print(np.ones(a, dtype=np.int))

###########################
###########################
import numpy as np

n, m = map(int, input().split())
print(np.eye(n, m))
# alternate
# print numpy.eye(*map(int,raw_input().split()))

###########################
###########################

import numpy as np

n, m = map(int, input().split())
a = np.array([[i for i in input().split()] for j in range(n)], int)
b = np.array([[i for i in input().split()] for j in range(n)], int)
print(np.add(a, b))
print(np.subtract(a, b))
print(np.multiply(a, b))
print(np.array(np.divide(a, b), int))
print(np.mod(a, b))
print(np.power(a, b))

a = numpy.array([1, 2, 3, 4], float)
b = numpy.array([5, 6, 7, 8], float)

print
a + b  # [  6.   8.  10.  12.]
print
numpy.add(a, b)  # [  6.   8.  10.  12.]

print
a - b  # [-4. -4. -4. -4.]
print
numpy.subtract(a, b)  # [-4. -4. -4. -4.]

print
a * b  # [  5.  12.  21.  32.]
print
numpy.multiply(a, b)  # [  5.  12.  21.  32.]

print
a / b  # [ 0.2         0.33333333  0.42857143  0.5       ]
print
numpy.divide(a, b)  # [ 0.2         0.33333333  0.42857143  0.5       ]

print
a % b  # [ 1.  2.  3.  4.]
print
numpy.mod(a, b)  # [ 1.  2.  3.  4.]

print
a ** b  # [  1.00000000e+00   6.40000000e+01   2.18700000e+03   6.55360000e+04]
print
numpy.power(a, b)  # [  1.00000000e+00   6.40000000e+01   2.18700000e+03   6.55360000e+04]
###########################
###########################
if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())
print[[i, j, k]
for i in range(x + 1) for j in range( y + 1) for k in range( z + 1) if ( ( i + j + k ) != n )]

###########################
###########################
# get second max number
if __name__ == '__main__':
    n = int(raw_input())
    # arr = map(int, raw_input().split())
    # print(sorted(list(set(arr)))[-2])
    arr = list(set(map(int, raw_input().split())))
    arr.remove(max(arr))
    print(max(arr))

###########################
###########################
lst = {}
for _ in range(int(raw_input())):
    name = raw_input()
    score = float(raw_input())
    lst[name] = score
print('\n'.join(
    sorted([x for x in lst if lst[x] == sorted(set([k[1] for k in sorted(lst.items(), key=lambda x: x[1])]))[1]])))
# n = int(input())
# marksheet = [[input(), float(input())] for _ in range(n)]

# arr = [[raw_input(), float(raw_input())] for _ in range(input())]
# c = sorted(set([b for a,b in arr]))[1]
# print '\n'.join(sorted([a for a,b in arr if b == c]))
###########################
###########################

if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
    print('%0.2f' % (sum(student_marks[query_name]) / len(student_marks[query_name])))

# marks = {}
# for _ in range(int(input())):
#    line = input().split()
#    marks[line[0]] = list(map(float, line[1:]))
# print('%.2f' %(sum(marks[input()])/3))
###########################
###########################
# get good csv
import os
import csv

goodfile = 0
# path = "/home/truename/python/sample/"#==43
path = "/home/truename/python/data/"
csvfiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith((".csv"))]
for ff in csvfiles:
    with open(ff) as f:
        reader = csv.reader(f)
        r = next(reader)
        if "pet" in [x.lower() for x in r]:
            # print(ff)
            colmn = [x.lower() for x in r].index("pet")
            # print(r,colmn)
            dog = 0
            cat = 0
            for row in reader:
                if "dog" == row[colmn].lower():
                    dog += 1
                if "cat" == row[colmn].lower():
                    cat += 1
            if dog > cat:
                goodfile += 1
print(goodfile)
# dog>cat

###########################
###########################
"""
Напишите программу, которая принимает на стандартный вход список игр футбольных команд с результатом матча и выводит на стандартный вывод сводную таблицу результатов всех матчей.

За победу команде начисляется 3 очка, за поражение — 0, за ничью — 1.

Формат ввода следующий:
В первой строке указано целое число n
— количество завершенных игр.
После этого идет n

строк, в которых записаны результаты игры в следующем формате:
Первая_команда;Забито_первой_командой;Вторая_команда;Забито_второй_командой

Вывод программы необходимо оформить следующим образом:
Команда:Всего_игр Побед Ничьих Поражений Всего_очков

Конкретный пример ввода-вывода приведён ниже.

Порядок вывода команд произвольный.

Sample Input:

3
Зенит;3;Спартак;1
Спартак;1;ЦСКА;1
ЦСКА;0;Зенит;2

Sample Output:

Зенит:2 2 0 0 6
ЦСКА:2 0 1 1 1
Спартак:2 0 1 1 1
"""
n = int(input())
d = {}
for i in range(n):
    x = input().split(';')
    if x[0] in d:
        d[x[0]]['cnt'] += 1
        d[x[0]]['w'] += int(int(x[1]) > int(x[3]))
        d[x[0]]['f'] += int(int(x[1]) < int(x[3]))
        d[x[0]]['o'] += int(int(x[1]) == int(x[3]))
    else:
        d[x[0]] = {'cnt': 1, 'w': int(int(x[1]) > int(x[3])), 'f': int(int(x[1]) < int(x[3])),
                   'o': int(int(x[1]) == int(x[3]))}
    if x[2] in d:
        d[x[2]]['cnt'] += 1
        d[x[2]]['w'] += int(int(x[3]) > int(x[1]))
        d[x[2]]['f'] += int(int(x[3]) < int(x[1]))
        d[x[2]]['o'] += int(int(x[3]) == int(x[1]))
    else:
        d[x[2]] = {'cnt': 1, 'w': int(int(x[3]) > int(x[1])), 'f': int(int(x[3]) < int(x[1])),
                   'o': int(int(x[3]) == int(x[1]))}
for i in d:
    print(i + ':' + str(d[i]['cnt']), d[i]['w'], d[i]['o'], d[i]['f'], (d[i]['w'] * 3 + d[i]['o']))

"""
В какой-то момент в Институте биоинформатики биологи перестали понимать, что говорят информатики: они говорили каким-то странным набором звуков. 

В какой-то момент один из биологов раскрыл секрет информатиков: они использовали при общении подстановочный шифр, т.е. заменяли каждый символ исходного сообщения на соответствующий ему другой символ. Биологи раздобыли ключ к шифру и теперь нуждаются в помощи: 

Напишите программу, которая умеет шифровать и расшифровывать шифр подстановки. Программа принимает на вход две строки одинаковой длины, на первой строке записаны символы исходного алфавита, на второй строке — символы конечного алфавита, после чего идёт строка, которую нужно зашифровать переданным ключом, и ещё одна строка, которую нужно расшифровать.

Пусть, например, на вход программе передано:
abcd
*d%#
abacabadaba
#*%*d*%

Это значит, что символ a исходного сообщения заменяется на символ * в шифре, b заменяется на d, c — на % и d — на #.
Нужно зашифровать строку abacabadaba и расшифровать строку #*%*d*% с помощью этого шифра. Получаем следующие строки, которые и передаём на вывод программы:
*d*%*d*#*d*
dacabac

Sample Input 1:

abcd
*d%#
abacabadaba
#*%*d*%

Sample Output 1:

*d*%*d*#*d*
dacabac


Sample Input 2:

dcba
badc
dcba
badc

Sample Output 2:

badc
dcba
"""
s0 = [i for i in input()]
s1 = [i for i in input()]
d = {}
for i in range(len(s0)):
    d[s0[i]] = s1[i]
for i in input():
    print(d[i], end='')
print()
for i in input():
    for j in d:
        if i == d[j]:
            print(j, end='')

"""
Простейшая система проверки орфографии основана на использовании списка известных слов. Каждое слово в проверяемом тексте ищется в этом списке и, если такое слово не найдено, оно помечается, как ошибочное.

Напишем подобную систему.

Через стандартный ввод подаётся следующая структура: первой строкой — количество d
записей в списке известных слов, после передаётся  d строк с одним словарным словом на строку, затем — количество l строк текста, после чего — l

строк текста.

Напишите программу, которая выводит слова из текста, которые не встречаются в словаре. Регистр слов не учитывается. Порядок вывода слов произвольный. Слова, не встречающиеся в словаре, не должны повторяться в выводе программы.

Sample Input:

3
a
bb
cCc
2
a bb aab aba ccc
c bb aaa

Sample Output:

aab
aba
c
aaa
"""
n = int(input())
d = []
for i in range(n):
    d.append(input().lower())
# t=[]
e = []
n = int(input())
for i in range(n):
    # t.append(input())
    for x in input().split():
        if (not x.lower() in d) and (not x.lower() in e):
            e.append(x.lower())
for i in e:
    print(i)

"""
Группа биологов в институте биоинформатики завела себе черепашку.

После дрессировки черепашка научилась понимать и запоминать указания биологов следующего вида:
север 10
запад 20
юг 30
восток 40
где первое слово — это направление, в котором должна двигаться черепашка, а число после слова — это положительное расстояние в сантиметрах, которое должна пройти черепашка.

Но команды даются быстро, а черепашка ползёт медленно, и программисты догадались, что можно написать программу, которая определит, куда в итоге биологи приведут черепашку. Для этого программисты просят вас написать программу, которая выведет точку, в которой окажется черепашка после всех команд. Для простоты они решили считать, что движение начинается в точке (0, 0), и движение на восток увеличивает первую координату, а на север — вторую.

Программе подаётся на вход число команд n
, которые нужно выполнить черепашке, после чего n

строк с самими командами. Вывести нужно два числа в одну строку: первую и вторую координату конечной точки черепашки. Все координаты целочисленные.

Sample Input:

4
север 10
запад 20
юг 30
восток 40

Sample Output:

20 -20
"""
n = int(input())
x = 0
y = 0
for i in range(n):
    p = input().split()
    if p[0] == 'север':
        y += int(p[1])
    elif p[0] == 'юг':
        y -= int(p[1])
    elif p[0] == 'восток':
        x += int(p[1])
    else:
        x -= int(p[1])
print(x, y)

"""
Дан файл с таблицей в формате TSV с информацией о росте школьников разных классов.

Напишите программу, которая прочитает этот файл и подсчитает для каждого класса средний рост учащегося.

Файл состоит из набора строк, каждая из которых представляет собой три поля:
Класс Фамилия Рост

Класс обозначается только числом. Буквенные модификаторы не используются. Номер класса может быть от 1 до 11 включительно. В фамилии нет пробелов, а в качестве роста используется натуральное число, но при подсчёте среднего требуется вычислить значение в виде вещественного числа.

Выводить информацию о среднем росте следует в порядке возрастания номера класса (для классов с первого по одиннадцатый). Если про какой-то класс нет информации, необходимо вывести напротив него прочерк, например:

Sample Input:

6	Вяххи	159
11	Федотов	172
7	Бондарев	158
6	Чайкина	153

Sample Output:

1 -
2 -
3 -
4 -
5 -
6 156.0
7 158.0
8 -
9 -
10 -
11 172.0
"""

"""
Напишите программу, которая подключает модуль math и, используя значение числа π

из этого модуля, находит для переданного ей на стандартный ввод радиуса круга периметр этого круга и выводит его на стандартный вывод.

Sample Input:

10.0

Sample Output:

62.83185307179586
"""
import math

print(2 * math.pi * float(input()))

"""
Напишите программу, которая запускается из консоли и печатает значения всех переданных аргументов на экран (имя скрипта выводить не нужно). Не изменяйте порядок аргументов при выводе.

Для доступа к аргументам командной строки программы подключите модуль sys и используйте переменную argv из этого модуля.

Пример работы программы:

> python3 my_solution.py arg1 arg2
arg1 arg2
"""
import sys

for i in range(1, len(sys.argv)):
    print(sys.argv[i], end=' ')

"""
Напишите функцию update_dictionary(d, key, value), которая принимает на вход словарь d и два числа: key и value

.

Если ключ key
есть в словаре d, то добавьте значение value в список, который хранится по этому ключу.
Если ключа key нет в словаре, то нужно добавить значение в список по ключу 2⋅key. Если и ключа 2⋅key нет, то нужно добавить ключ 2⋅key в словарь и сопоставить ему список из переданного элемента [value]

.

Требуется реализовать только эту функцию, кода вне неё не должно быть.
Функция не должна вызывать внутри себя функции input и print.

Пример работы функции:

d = {}
print(update_dictionary(d, 1, -1))  # None
print(d)                            # {2: [-1]}
update_dictionary(d, 2, -2)
print(d)                            # {2: [-1, -2]}
update_dictionary(d, 1, -3)
print(d)                            # {2: [-1, -2, -3]}
"""


# не добавляйте кода вне функции
def update_dictionary(d, key, value):
    # put your python code here
    if key in d:
        d[key].append(value)
    elif (2 * key) in d:
        d[(key * 2)].append(value)
    else:
        d[(key * 2)] = [value]


# не добавляйте кода вне функции

"""
Когда Антон прочитал «Войну и мир», ему стало интересно, сколько слов и в каком количестве используется в этой книге.

Помогите Антону написать упрощённую версию такой программы, которая сможет подсчитать слова, разделённые пробелом и вывести получившуюся статистику.

Программа должна считывать одну строку со стандартного ввода и выводить для каждого уникального слова в этой строке число его повторений (без учёта регистра) в формате "слово количество" (см. пример вывода).
Порядок вывода слов может быть произвольным, каждое уникальное слово должно выводиться только один раз.

Sample Input 1:

a aa abC aa ac abc bcd a

Sample Output 1:

ac 1
a 2
abc 2
bcd 1
aa 2


Sample Input 2:

a A a

Sample Output 2:

a 3
"""
# put your python code here
s = input()
d = {}
for i in s.split():
    if i.lower() in d:
        d[i.lower()] += 1
    else:
        d[i.lower()] = 1
for i in d:
    print(i, d[i])

"""
Имеется реализованная функция f(x), принимающая на вход целое число x

, которая вычисляет некоторое целочисленое значение и возвращает его в качестве результата работы.

Функция вычисляется достаточно долго, ничего не выводит на экран, не пишет в файлы и зависит только от переданного аргумента x

.

Напишите программу, которой на вход в первой строке подаётся число n
— количество значений x, для которых требуется узнать значение функции f(x), после чего сами эти n значений, каждое на отдельной строке. Программа должна после каждого введённого значения аргумента вывести соответствующие значения функции f

 на отдельной строке. 

Для ускорения вычисления необходимо сохранять уже вычисленные значения функции при известных аргументах.

Обратите внимание, что в этой задаче установлено достаточно сильное ограничение в две секунды по времени исполнения кода на тесте. 

Sample Input:

5
5
12
9
20
12

Sample Output:

11
41
47
61
41
"""
# Считайте, что функция f(x) уже определена выше. Определять её отдельно не требуется.
n = int(input())
d = {}
for i in range(n):
    k = int(input())
    if k in d:
        print(d[k])
    else:
        v = f(k)
        d[k] = v
        print(v)
"""
Напишите функцию f(x), которая возвращает значение следующей функции, определённой на всей числовой прямой:

f(x)=⎧⎩⎨⎪⎪1−(x+2)2,−x2,(x−2)2+1,при x≤−2при −2<x≤2при 2<x

Требуется реализовать только функцию, решение не должно осуществлять операций ввода-вывода.

Sample Input 1:

4.5

Sample Output 1:

7.25


Sample Input 2:

-4.5

Sample Output 2:

-5.25


Sample Input 3:

1

Sample Output 3:

-0.5
"""


def f(x):
    if x <= -2:
        return 1 - (x + 2) ** 2
    elif -2 < x <= 2:
        return -x / 2
    else:
        return (x - 2) ** 2 + 1


"""


Напишите функцию modify_list(l), которая принимает на вход список целых чисел, удаляет из него все нечётные значения, а чётные нацело делит на два. Функция не должна ничего возвращать, требуется только изменение переданного списка, например:

lst = [1, 2, 3, 4, 5, 6]
print(modify_list(lst))  # None
print(lst)               # [1, 2, 3]
modify_list(lst)
print(lst)               # [1]

lst = [10, 5, 8, 3]
modify_list(lst)
print(lst)               # [5, 4]

Функция не должна осуществлять ввод/вывод информации.

"""


def modify_list(l):
    for i in range(len(l)):
        if l[i] % 2 == 1:
            l[i] = -1
    while -1 in l:
        l.remove(-1)
    for i in range(len(l)):
        l[i] = l[i] // 2


"""
Напишите программу, которая считывает со стандартного ввода целые числа, по одному числу в строке, и после первого введенного нуля выводит сумму полученных на вход чисел.

Sample Input 1:

5
-3
8
4
0

Sample Output 1:

14


Sample Input 2:

0

Sample Output 2:

0
"""
n = int(input())
s = 0
while n != 0:
    s += n
    n = int(input())
print(s)

"""
В Институте биоинформатики между информатиками и биологами устраивается соревнование. Победителям соревнования достанется большой и вкусный пирог. В команде биологов a человек, а в команде информатиков — b

человек. 

Нужно заранее разрезать пирог таким образом, чтобы можно было раздать кусочки пирога любой команде, выигравшей соревнование, при этом каждому участнику этой команды должно достаться одинаковое число кусочков пирога. И так как не хочется резать пирог на слишком мелкие кусочки, нужно найти минимальное подходящее число.

Напишите программу, которая помогает найти это число.
Программа должна считывать размеры команд (два положительных целых числа a
и b, каждое число вводится на отдельной строке) и выводить наименьшее число d

, которое делится на оба этих числа без остатка.


Sample Input 1:

7
5

Sample Output 1:

35


Sample Input 2:

15
15

Sample Output 2:

15


Sample Input 3:

12
16

Sample Output 3:

48
"""
a = int(input())
b = int(input())
n = 1
while not (n % a == 0 and n % b == 0):
    n += 1
print(n)
"""
Напишите программу, которая считывает целые числа с консоли по одному числу в строке.

Для каждого введённого числа проверить:
если число меньше 10, то пропускаем это число;
если число больше 100, то прекращаем считывать числа;
в остальных случаях вывести это число обратно на консоль в отдельной строке.

Sample Input 1:

12
4
2
58
112

Sample Output 1:

12
58


Sample Input 2:

101

Sample Output 2:


Sample Input 3:

1
2
102

Sample Output 3: 
"""
a = 0
while a >= 0:
    a = int(input())
    if a < 10:
        continue
    if a > 100:
        break
    print(a)
"""
Когда Павел учился в школе, он запоминал таблицу умножения прямоугольными блоками. Для тренировок ему бы очень пригодилась программа, которая показывала бы блок таблицы умножения.

Напишите программу, на вход которой даются четыре числа a
, b, c и d, каждое в своей строке. Программа должна вывести фрагмент таблицы умножения для всех чисел отрезка [a;b] на все числа отрезка [c;d]

.

Числа a
, b, c и d являются натуральными и не превосходят 10, a≤b, c≤d

.

Следуйте формату вывода из примера, для разделения элементов внутри строки используйте '\t' — символ табуляции. Заметьте, что левым столбцом и верхней строкой выводятся сами числа из заданных отрезков — заголовочные столбец и строка таблицы.

Sample Input 1:

7
10
5
6

Sample Output 1:

	5	6
7	35	42
8	40	48
9	45	54
10	50	60


Sample Input 2:

5
5
6
6

Sample Output 2:

	6
5	30


Sample Input 3:

1
3
2
4

Sample Output 3:

	2	3	4
1	2	3	4
2	4	6	8
3	6	9	12
"""
a = int(input())
b = int(input())
c = int(input())
d = int(input())
print('', end='\t'),
for i in range(c, d + 1):
    print(i, end='\t')
print()
for i in range(a, b + 1):
    print(i, end='\t')
    for j in range(c, d + 1):
        print(j * i, end='\t')
    print()

"""
Напишите программу, которая считывает с клавиатуры два числа a и b, считает и выводит на консоль среднее арифметическое всех чисел из отрезка [a;b], которые делятся на 3

.

В приведенном ниже примере среднее арифметическое считается для чисел на отрезке [−5;12]
. Всего чисел, делящихся на 3, на этом отрезке 6: −3,0,3,6,9,12. Их среднее арифметическое равно 4.5

На вход программе подаются интервалы, внутри которых всегда есть хотя бы одно число, которое делится на 3

.

Sample Input:

-5
12

Sample Output:

4.5
"""
a = int(input())
b = int(input())
s = 0
c = 0
for i in range(a, b + 1):
    if i % 3 == 0:
        s += i
        c += 1
print(s / c)

"""
GC-состав является важной характеристикой геномных последовательностей и определяется как процентное соотношение суммы всех гуанинов и цитозинов к общему числу нуклеиновых оснований в геномной последовательности.

Напишите программу, которая вычисляет процентное содержание символов G (гуанин) и C (цитозин) в введенной строке (программа не должна зависеть от регистра вводимых символов).

Например, в строке "acggtgttat" процентное содержание символов G и C равно 410⋅100=40.0

, где 4 -- это количество символов G и C,  а 10 -- это длина строки.

Sample Input:

acggtgttat

Sample Output:

40.0
"""
s = input()
print((s.lower().count('c') + s.lower().count('g')) * 100 / len(s))
"""
Узнав, что ДНК не является случайной строкой, только что поступившие в Институт биоинформатики студенты группы информатиков предложили использовать алгоритм сжатия, который сжимает повторяющиеся символы в строке.

Кодирование осуществляется следующим образом:
s = 'aaaabbсaa' преобразуется в 'a4b2с1a2', то есть группы одинаковых символов исходной строки заменяются на этот символ и количество его повторений в этой позиции строки.

Напишите программу, которая считывает строку, кодирует её предложенным алгоритмом и выводит закодированную последовательность на стандартный вывод. Кодирование должно учитывать регистр символов.

Sample Input 1:

aaaabbcaa

Sample Output 1:

a4b2c1a2


Sample Input 2:

abc

Sample Output 2:

a1b1c1
"""
s = input()
p = s[0]
i = 1
r = ''
for c in s[1:-1]:
    if c == p:
        i += 1
    else:
        r += p + str(i)
        i = 1
    p = c
if len(s) > 1:
    if s[-1] == p:
        i += 1
    else:
        r += p + str(i)
        i = 1
r += s[-1] + str(i)
print(r)

"""
Напишите программу, на вход которой подается одна строка с целыми числами. Программа должна вывести сумму этих чисел.

Используйте метод split строки.

Sample Input:

4 -1 9 3

Sample Output:

15
"""
s = 0
for i in input().split():
    s += int(i)
print(s)
"""
Напишите программу, на вход которой подаётся список чисел одной строкой. Программа должна для каждого элемента этого списка вывести сумму двух его соседей. Для элементов списка, являющихся крайними, одним из соседей считается элемент, находящий на противоположном конце этого списка. Например, если на вход подаётся список "1 3 5 6 10", то на выход ожидается список "13 6 9 15 7" (без кавычек).

Если на вход пришло только одно число, надо вывести его же.

Вывод должен содержать одну строку с числами нового списка, разделёнными пробелом.

Sample Input 1:

1 3 5 6 10

Sample Output 1:

13 6 9 15 7


Sample Input 2:

10

Sample Output 2:

10
"""
s = input()
a = s.split()
b = []
if len(a) == 1:
    print(s)
else:
    for i in range(len(a) - 1):
        b.append(int(a[i - 1]) + int(a[i + 1]))
    b.append(int(a[0]) + int(a[len(a) - 2]))
    for n in b:
        print(n, end=' ')

"""
Напишите программу, которая принимает на вход список чисел в одной строке и выводит на экран в одну строку значения, которые повторяются в нём более одного раза.

Для решения задачи может пригодиться метод sort списка.

Выводимые числа не должны повторяться, порядок их вывода может быть произвольным.

Sample Input 1:

4 8 0 3 4 2 0 3

Sample Output 1:

0 3 4


Sample Input 2:

10

Sample Output 2:


Sample Input 3:

1 1 2 2 3 3

Sample Output 3:

1 2 3


Sample Input 4:

1 1 1 1 1 2 2 2

Sample Output 4:

1 2
"""
s = input()
a = [int(i) for i in sorted(s.split())]
b = []
if len(a) == 1:
    print('')
else:
    for i in range(1, len(a)):
        if a[i - 1] == a[i] and a[i] not in b:
            b.append(a[i - 1])
    if len(b) > 0:
        for n in b:
            print(n, end=' ')
    else:
        print('')

"""
Напишите программу, которая считывает с консоли числа (по одному в строке) до тех пор, пока сумма введённых чисел не будет равна 0 и сразу после этого выводит сумму квадратов всех считанных чисел.

Гарантируется, что в какой-то момент сумма введённых чисел окажется равной 0, после этого считывание продолжать не нужно.

В примере мы считываем числа 1, -3, 5, -6, -10, 13; в этот момент замечаем, что сумма этих чисел равна нулю и выводим сумму их квадратов, не обращая внимания на то, что остались ещё не прочитанные значения.

Sample Input:

1
-3
5
-6
-10
13
4
-8

Sample Output:

340
"""
s = int(input())
p = s * s
while s != 0:
    i = int(input())
    s += i
    p += i * i
print(p)
"""
Напишите программу, которая выводит часть последовательности 1 2 2 3 3 3 4 4 4 4 5 5 5 5 5 ... (число повторяется столько раз, чему равно). На вход программе передаётся положительное целое число n — столько элементов последовательности должна отобразить программа. На выходе ожидается последовательность чисел, записанных через пробел в одну строку. 

Например, если n = 7, то программа должна вывести 1 2 2 3 3 3 4.

Sample Input:

7

Sample Output:

1 2 2 3 3 3 4
"""
n = int(input())
b = []
i = 1
while i <= n:
    for c in range(i):
        b.append(i)
    i += 1
for i in range(n):
    print(b[i], end=' ')

"""
Напишите программу, которая считывает список чисел lst из первой строки и число x из второй строки, которая выводит все позиции, на которых встречается число x в переданном списке lst

.

Позиции нумеруются с нуля, если число x

не встречается в списке, вывести строку "Отсутствует" (без кавычек, с большой буквы).

Позиции должны быть выведены в одну строку, по возрастанию абсолютного значения.

Sample Input 1:

5 8 2 7 8 8 2 4
8

Sample Output 1:

1 4 5


Sample Input 2:

5 8 2 7 8 8 2 4
10

Sample Output 2:

Отсутствует
"""
l = [int(i) for i in input().split()]
n = int(input())
b = []
for i in range(len(l)):
    if l[i] == n:
        b.append(i)
if len(b) == 0:
    print("Отсутствует")
else:
    for i in b:
        print(i, end=' ')

"""
Напишите программу, на вход которой подаётся прямоугольная матрица в виде последовательности строк, заканчивающихся строкой, содержащей только строку "end" (без кавычек)

Программа должна вывести матрицу того же размера, у которой каждый элемент в позиции i, j равен сумме элементов первой матрицы на позициях (i-1, j), (i+1, j), (i, j-1), (i, j+1). У крайних символов соседний элемент находится с противоположной стороны матрицы.

В случае одной строки/столбца элемент сам себе является соседом по соответствующему направлению.

Sample Input 1:

9 5 3
0 7 -1
-5 2 9
end

Sample Output 1:

3 21 22
10 6 19
20 16 -1


Sample Input 2:

1
end

Sample Output 2:

4
"""
s = input()
b = []
c = []
while s != 'end':
    b.append([int(i) for i in s.split()])
    c.append([int(i) for i in s.split()])
    s = input()
# print(len(b))
# print(len(b[0]))
for i in range(len(b)):
    for j in range(len(b[i])):
        if (i == (len(b) - 1)) and (j != (len(b[i]) - 1)):
            # print('a',i,j,b[i-1][j]+b[0][j]+b[i][j-1]+b[i][j+1])
            c[i][j] = b[i - 1][j] + b[0][j] + b[i][j - 1] + b[i][j + 1]
        elif (i != (len(b) - 1)) and (j == (len(b[i]) - 1)):
            # print('b',i,j,b[i-1][j]+b[i+1][j]+b[i][j-1]+b[i][0])
            c[i][j] = b[i - 1][j] + b[i + 1][j] + b[i][j - 1] + b[i][0]
        elif (i == (len(b) - 1)) and (j == (len(b[i]) - 1)):
            # print('c',i,j,b[i-1][j]+b[0][j]+b[i][j-1]+b[i][0])
            c[i][j] = b[i - 1][j] + b[0][j] + b[i][j - 1] + b[i][0]
        else:
            # print('d',i,j,b[i-1][j]+b[i+1][j]+b[i][j-1]+b[i][j+1])
            c[i][j] = b[i - 1][j] + b[i + 1][j] + b[i][j - 1] + b[i][j + 1]
for i in c:
    for j in i:
        print(j, end=' ')
    print()
"""
Выведите таблицу размером n×n, заполненную числами от 1 до n2 по спирали, выходящей из левого верхнего угла и закрученной по часовой стрелке, как показано в примере (здесь n=5):

Sample Input:

5

Sample Output:

1 2 3 4 5
16 17 18 19 6
15 24 25 20 7
14 23 22 21 8
13 12 11 10 9
"""
n = int(input())
a = [[0 for i in range(n)] for j in range(n)]
# b=[i for i in range(1,n*n+1)]
# print(a)
r = 0
c = 0
k = 1
j = 0
while k <= n * n:
    for c in range(j, n - j):
        a[r][c] = k
        # print(r,c,k)
        k += 1
    r += 1
    for rr in range(r, n - r + 1):
        a[rr][c] = k
        # print(rr,c,k)
        k += 1
    j += 1
    for c in range(n - j - 1, j - 1 - 1, -1):
        a[rr][c] = k
        # print(rr,c,k)
        k += 1
    for rr in range(n - r - 1, r - 1, -1):
        a[rr][c] = k
        # print(rr,c,k)
        k += 1
for i in range(n):
    for j in range(n):
        print(a[i][j], end=' ')
    print()

"""
В то далёкое время, когда Паша ходил в школу, ему очень не нравилась формула Герона для вычисления площади треугольника, так как казалась слишком сложной. В один прекрасный момент Павел решил избавить всех школьников от страданий и написать и распространить по школам программу, вычисляющую площадь треугольника по трём сторонам. 

Одна проблема: так как эта формула не нравилась Павлу, он её не запомнил. Помогите ему завершить доброе дело и напишите программу, вычисляющую площадь треугольника по переданным длинам трёх его сторон по формуле Герона:
S=p(p−a)(p−b)(p−c)‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾√

где p=a+b+c2

– полупериметр треугольника. На вход программе подаются целые числа, выводом программы должно являться вещественное число, соответствующее площади треугольника.

Sample Input:

3
4
5

Sample Output:

6.0
"""
a = int(input())
b = int(input())
c = int(input())
p = (a + b + c) / 2
s = (p * (p - a) * (p - b) * (p - c)) ** (1 / 2)
print(s)
"""
Напишите программу, принимающую на вход целое число, которая выводит True, если переданное значение попадает в интервал (−15,12]∪(14,17)∪[19,+∞) и False в противном случае (регистр символов имеет значение).

Обратите внимание на разные скобки, используемые для обозначения интервалов. В задании используются полуоткрытые и открытые интервалы. Подробнее про это вы можете прочитать, например, на википедии (полуинтервал, промежуток).

Sample Input 1:

20

Sample Output 1:

True


Sample Input 2:

-20

Sample Output 2:

False
"""
a = int(input())
print((-15 < a <= 12) or (14 < a < 17) or (a >= 19))
"""
Напишите простой калькулятор, который считывает с пользовательского ввода три строки: первое число, второе число и операцию, после чего применяет операцию к введённым числам ("первое число" "операция" "второе число") и выводит результат на экран.

Поддерживаемые операции: +, -, /, *, mod, pow, div, где
mod — это взятие остатка от деления,
pow — возведение в степень,
div — целочисленное деление.

Если выполняется деление и второе число равно 0, необходимо выводить строку "Деление на 0!".

Обратите внимание, что на вход программе приходят вещественные числа.

Sample Input 1:

5.0
0.0
mod

Sample Output 1:

Деление на 0!


Sample Input 2:

-12.0
-8.0
*

Sample Output 2:

96.0


Sample Input 3:

5.0
10.0
/

Sample Output 3:

0.5
"""
a = float(input())
b = float(input())
c = input()
if c == "+":
    print(a + b)
elif c == "-":
    print(a - b)
elif c == "/":
    if b == 0:
        print('Деление на 0!')
    else:
        print(a / b)
elif c == '*':
    print(a * b)
elif c == 'mod':
    if b == 0:
        print('Деление на 0!')
    else:
        print(a % b)
elif c == 'pow':
    if b == -1:
        print('Деление на 0!')
    else:
        print(a ** b)
elif c == 'div':
    if int(b) == 0:
        print('Деление на 0!')
    else:
        print(a // b)
"""
Жители страны Малевии часто экспериментируют с планировкой комнат. Комнаты бывают треугольные, прямоугольные и круглые. Чтобы быстро вычислять жилплощадь, требуется написать программу, на вход которой подаётся тип фигуры комнаты и соответствующие параметры, которая бы выводила площадь получившейся комнаты.
Для числа π в стране Малевии используют значение 3.14.

Формат ввода, который используют Малевийцы:

треугольник
a
b
c

где a, b и c — длины сторон треугольника

прямоугольник
a
b

где a и b — длины сторон прямоугольника

круг
r

где r — радиус окружности

Sample Input 1:

прямоугольник
4
10

Sample Output 1:

40.0


Sample Input 2:

круг
5

Sample Output 2:

78.5


Sample Input 3:

треугольник
3
4
5

Sample Output 3:

6.0
"""
t = input()
if t == 'треугольник':
    a = float(input())
    b = float(input())
    c = float(input())
    p = (a + b + c) / 2
    s = (p * (p - a) * (p - b) * (p - c)) ** (1 / 2)
elif t == 'прямоугольник':
    a = float(input())
    b = float(input())
    s = a * b
else:
    r = float(input())
    s = 3.14 * r * r
print(s)
"""
Напишите программу, которая получает на вход три целых числа, по одному числу в строке, и выводит на консоль в три строки сначала максимальное, потом минимальное, после чего оставшееся число.

На ввод могут подаваться и повторяющиеся числа.

Sample Input 1:

8
2
14

Sample Output 1:

14
2
8


Sample Input 2:

23
23
21

Sample Output 2:

23
21
23
"""
a = int(input())
b = int(input())
c = int(input())
max_ = max(a, b, c)
min_ = min(a, b, c)
print(max_)
print(min_)
if b <= a <= c:
    print(a)
elif c <= a <= b:
    print(a)
elif a <= b <= c:
    print(b)
elif c <= b <= a:
    print(b)
elif a <= c <= b:
    print(c)
elif b <= c <= a:
    print(c)
"""
В институте биоинформатики по офису передвигается робот. Недавно студенты из группы программистов написали для него программу, по которой робот, когда заходит в комнату, считает количество программистов в ней и произносит его вслух: "n программистов".

Для того, чтобы это звучало правильно, для каждого n

нужно использовать верное окончание слова.

Напишите программу, считывающую с пользовательского ввода целое число n

(неотрицательное), выводящее это число в консоль вместе с правильным образом изменённым словом "программист", для того, чтобы робот мог нормально общаться с людьми, например: 1 программист, 2 программиста, 5 программистов.

В комнате может быть очень много программистов. Проверьте, что ваша программа правильно обработает все случаи, как минимум до 1000 человек.

Дополнительный комментарий к условию:
Обратите внимание, что задача не так проста, как кажется на первый взгляд. Если ваше решение не проходит какой-то тест, это значит, что вы не рассмотрели какой-то из случаев входных данных (число программистов 0≤n≤1000

). Обязательно проверяйте свои решения на дополнительных значениях, а не только на тех, что приведены в условии задания. 

Так как задание повышенной сложности, вручную код решений проверяться не будет. Если вы столкнулись с ошибкой в первых четырёх тестах, проверьте, что вы используете только русские символы для ответа. В остальных случаях ищите ошибку в логике работы программы.

Sample Input 1:

5

Sample Output 1:

5 программистов


Sample Input 2:

0

Sample Output 2:

0 программистов


Sample Input 3:

1

Sample Output 3:

1 программист


Sample Input 4:

2

Sample Output 4:

2 программиста
"""
n = int(input())
if n % 10 == 0 or n % 10 >= 5 or 10 <= n <= 20 or 11 <= n % 100 <= 20:
    print(n, 'программистов')
elif n % 10 == 1:
    print(n, 'программист')
elif 2 <= n % 10 <= 4:
    print(n, 'программиста')
"""
Паша очень любит кататься на общественном транспорте, а получая билет, сразу проверяет, счастливый ли ему попался. Билет считается счастливым, если сумма первых трех цифр совпадает с суммой последних трех цифр номера билета.

Однако Паша очень плохо считает в уме, поэтому попросил вас написать программу, которая проверит равенство сумм и выведет "Счастливый", если суммы совпадают, и "Обычный", если суммы различны.

На вход программе подаётся строка из шести цифр.

Выводить нужно только слово "Счастливый" или "Обычный", с большой буквы.

Sample Input 1:

090234

Sample Output 1:

Счастливый


Sample Input 2:

123456

Sample Output 2:

Обычный
"""
n = int(input())
if n % 10 + n % 100 // 10 + n % 1000 // 100 == n % 10000 // 1000 + n % 100000 // 10000 + n // 100000:
    print('Счастливый')
else:
    print('Обычный')

###########################
###########################
# https://ideone.com/KGYDob

import sys

prev = ''
avert = []
for line in sys.stdin:
    vert, dist, vs = line.split()
    if (vert != prev) and (prev != ''):
        print(prev + '\t' + "{:.3f}".format(0.02 + 0.9 * avert[0]) + '\t' + avert[1])
        if vs == '{}':
            avert = [float(dist), vs]
        else:
            avert = [0, vs]
        prev = vert
    else:
        prev = vert
        if len(avert) > 0:
            if vs == '{}':
                avert[0] += float(dist)
            else:
                avert = [avert[0], vs]
        else:
            if vs == '{}':
                avert = [float(dist), vs]
            else:
                avert = [0, vs]
print(prev + '\t' + "{:.3f}".format(0.02 + 0.9 * avert[0]) + '\t' + avert[1])

'''
1	0.067	{}
1	0.200	{2,4}
2	0.067	{}
2	0.100	{}
2	0.200	{3,5}
3	0.067	{}
3	0.100	{}
3	0.200	{4}
4	0.100	{}
4	0.200	{}
4	0.200	{5}
5	0.100	{}
5	0.200	{}
5	0.200	{1,2,3}
'''

v = int(input())
t = int(input())
if v >= 0:
    l = (v * t) % 109
else:
    if t > 0:
        l = 109 + ((v * t) % (-109))
    else:
        l = 0
if l = 109:
    l = 0
print(str(l))

h1 = int(input())
m1 = int(input())
s1 = int(input())
h2 = int(input())
m2 = int(input())
s2 = int(input())
c = (h2 * 3600 + m2 * 60 + s2) - (h1 * 3600 + m1 * 60 + s1)
print(str(c))

n = int(input())
h = (n // 3600) % 24
m = (n % 3600) // 60
s = (n % 3600) % 60
print(str(h) + ':' + "{:02d}".format(m) + ':' + "{:02d}".format(s))

a = int(input())
b = int(input())
n = int(input())
c = a * 100 + b
na = (c * n) // 100
nb = (c * n) % 100
print(str(na) + ' ' + str(nb))

###########################
###########################
n = int(input())
k1a = [i for i in input().split()]
k1 = k1a[0]
k1a.pop(0)
k2a = [i for i in input().split()]  # map(int, input().split())
k2 = k2a[0]
k2a.pop(0)
x = 0
while k1a and k2a:
    x += 1
    a, b = k1a.pop(0), k2a.pop(0)
    # print(x,a,b,k1a,k2a)
    if a > b:
        # print('a>b')
        k1a += [b, a]
    else:
        k2a += [a, b]
    if x == 1000000:
        print(-1)
        break
if x != 1000000:
    print(x, 1 if k1a else 2)

###########################
###########################
import os
import csv

goodfile = 0
# path = "/home/truename/python/sample/"#==43
path = "/home/truename/python/data/"
csvfiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith((".csv"))]
for ff in csvfiles:
    with open(ff) as f:
        reader = csv.reader(f)
        r = next(reader)
        if "pet" in [x.lower() for x in r]:
            # print(ff)
            colmn = [x.lower() for x in r].index("pet")
            # print(r,colmn)
            dog = 0
            cat = 0
            for row in reader:
                if "dog" == row[colmn].lower():
                    dog += 1
                if "cat" == row[colmn].lower():
                    cat += 1
            if dog > cat:
                goodfile += 1
print(goodfile)
# dog>cat


###########################
###########################
""" 
Напишите программу, которая выводит часть последовательности 1 2 2 3 3 3 4 4 4 4 5 5 5 5 5 ... (число повторяется столько раз, 
чему равно). 
На вход программе передаётся положительное целое число n — столько элементов последовательности должна отобразить программа. 
На выходе ожидается последовательность чисел, записанных через пробел в одну строку. 

Например, если n = 7, то программа должна вывести 1 2 2 3 3 3 4. 
  
Sample Input: 
7 

Sample Output: 
1 2 2 3 3 3 4 

n=int(input()) 
b=[] 
i=1 
while i<=n: 
        for c in range(i): 
                b.append(i) 
        i+=1 
for i in range(n): 
        print(b[i],end=' ') 
"""

""" 
Напишите программу, которая считывает список чисел lst из первой строки и число x 
 из второй строки, которая выводит все позиции, на которых встречается число x 
 в переданном списке lst. 

Позиции нумеруются с нуля, если число x 
 не встречается в списке, вывести строку "Отсутствует" (без кавычек, с большой буквы). 
Позиции должны быть выведены в одну строку, по возрастанию абсолютного значения. 

Sample Input 1: 
5 8 2 7 8 8 2 4 
8 
Sample Output 1: 
1 4 5 

Sample Input 2: 
5 8 2 7 8 8 2 4 
10 
Sample Output 2: 
Отсутствует 

l=[int(i) for i in input().split()] 
n=int(input()) 
b=[] 
for i in range(len(l)): 
        if l[i]==n: 
                b.append(i) 
if len(b)==0: 
        print("Отсутствует") 
else: 
        for i in b: 
                print(i,end=' ') 
"""
""" 
Напишите программу, на вход которой подаётся прямоугольная матрица в виде последовательности строк, 
заканчивающихся строкой, содержащей только строку "end" (без кавычек) 
Программа должна вывести матрицу того же размера, у которой каждый элемент в позиции i, j равен 
сумме элементов первой матрицы на позициях (i-1, j), (i+1, j), (i, j-1), (i, j+1). 
У крайних символов соседний элемент находится с противоположной стороны матрицы. 

В случае одной строки/столбца элемент сам себе является соседом по соответствующему направлению. 

Sample Input 1: 
9 5 3 
0 7 -1 
-5 2 9 
end 

Sample Output 1: 
3 21 22 
10 6 19 
20 16 -1 

Sample Input 2: 
1 
end 

Sample Output 2: 
4 
"""
""" 
n=int(input()) 
a=[[0 for i in range(n)] for j in range(n)] 
b=[i for i in range(1,n*n+1)] 
print(a) 
k=1 
f=1 
j=0 
r=0 
while k<=n*n: 
        for cc in range(j,n-j): 
                a[r][cc]=k 
        k+=1 
        r+=1 
        for rr in range(r,n): 
                a[rr][cc]=k 
        k+=1 
        c+=1 
        for cc in range(j,n-j): 
                a[r][cc]=k 
        k+=1 
        r+=1 
------------------------------------- 
"""
""" 
def modify_list(l): 
        for i in range(len(l)): 
                if l[i]%2==1: 
                        l[i]=-1 
        while -1 in l: 
                l.remove(-1) 
        for i in range(len(l)): 
            l[i]=l[i]//2 

lst = [1, 2, 3, 3, 5, 4, 5, 7, 6] 
modify_list(lst) 
print(lst) 
"""

""" 
Реализуйте программу, которая будет вычислять количество различных объектов в списке. 
Два объекта a и b считаются различными, если a is b равно False. 
Вашей программе доступна переменная с названием objects, которая ссылается на список, содержащий не более 100 объектов. Выведите количество различных объектов в этом списке. 

Формат ожидаемой программы: 
ans = 0 
for obj in objects: # доступная переменная objects 
    ans += 1 

print(ans) 

Примечание: 
Количеством различных объектов называется максимальный размер множества объектов, в котором любые два объекта являются различными. 

Рассмотрим пример: 
objects = [1, 2, 1, 2, 3] # будем считать, что одинаковые числа соответствуют одинаковым объектам, а различные – различным 
Тогда все различные объекты являют собой множество {1, 2, 3}﻿. Таким образом, количество различных объектов равно трём. 

objects=input() 
ans = 0 
b=[] 
for obj in objects: 
        if obj not in b: 
                b.append(obj) 
                ans += 1 
print(ans) 
"""

""" 
Напишите реализацию функции closest_mod_5, принимающую в качестве единственного аргумента целое число x и возвращающую самое маленькое целое число y, такое что: 

•y больше или равно x 

•y делится нацело на 5 

Формат того, что ожидается от вас в качестве ответа: 


def closest_mod_5(x): 
    if x % 5 == 0: 
        return x 
    return "I don't know :(" 

def closest_mod_5(x): 
        if x % 5 == 0: 
                return x 
        else: 
                return closest_mod_5(x+1) 
        return "I don't know :(" 
a=int(input()) 
print(closest_mod_5(a)) 
"""
""" 
def s(a, *vs, b=10): 
   res = a + b 
   for v in vs: 
       res += v 
   return res 
#print(s(b=31, 0)) 
print(s(11, 10, 10)) 
print(s(11, 10, b=10)) 
print(s(5, 5, 5, 5, 1)) 
#print(s(b=31)) 
print(s(11, 10)) 
print(s(21)) 
print(s(11, b=20)) 
print(s(0, 0, 31)) 
"""
""" 
def getIsSubClass(c1,c2): 
        b=False 
        if c1==c2: 
                return True 
        else: 
                if len(s[c2]['pid'])==0: 
                        return False 
                else: 
                        if c1 in s[c2]['pid']: 
                                #b=b or True 
                                return True 
                        else: 
                                for j in s[c2]['pid']: 
                                        b=b or getIsSubClass(c1,j) 
                                return b 
n=int(input()) 
s={} 
for i in range(n): 
        x=[i for i in input().split()] 
        if ":" in x: 
                s[x[0]]={'pid':x[2:]} 
                for k in x[2:]: 
                        if k not in s: 
                                s[k]={'pid':[]} 
        else: 
                s[x[0]]={'pid':[]} 
#print(s) 
q=int(input()) 
for i in range(q): 
        a,b=input().split() 
        if getIsSubClass(a,b): 
                print('Yes') 
        else: 
                print('No') 
"""
""" 
Одно из применений множественного наследование – расширение функциональности класса каким-то заранее определенным способом. Например, если нам понадобится логировать какую-то информацию при обращении к методам класса. 

Рассмотрим класс Loggable: 



import time 

class Loggable: 
    def log(self, msg): 
        print(str(time.ctime()) + ": " + str(msg)) 

У него есть ровно один метод log, который позволяет выводить в лог (в данном случае в stdout) какое-то сообщение, добавляя при этом текущее время. 


Реализуйте класс LoggableList, отнаследовав его от классов list и Loggable таким образом, чтобы при добавлении элемента в список посредством метода append в лог отправлялось сообщение, состоящее из только что добавленного элемента. 




Примечание 
Ваша программа не должна содержать класс Loggable. При проверке вашей программе будет доступен этот класс, и он будет содержать метод log﻿, описанный выше. 
"""
""" 
def getmax(l): 
        d={} 
        for i in l.split(): 
                if i.lower() in d: 
                        d[i.lower()]['cnt']+=1 
                else: 
                        d[i.lower()]={'cnt':1,'wrd':i.lower()} 
        print(d) 
        max=0 
        maxs='' 
        for i in d: 
                if d[i]['cnt']>max: 
                        max=d[i]['cnt'] 
                        maxs=i 
                elif d[i]['cnt']==max and i<maxs: 
                        max=d[i]['cnt'] 
                        maxs=i 
        print(maxs,max) 
testfile = open("test.txt") 
s=testfile.read(); 
getmax(s) 
testfile.close() 
"""
""" 
def getmarks(l): 
        d={} 
        d['sum']={1:0,2:0,3:0} 
        for line in l: 
                p=line.replace('\n', '').split(';') 
                #print(p) 
                d[p[0]]={1:int(p[1]),2:int(p[2]),3:int(p[3]),'avg':(int(p[1])+int(p[2])+int(p[3]))/3} 
                print(d[p[0]]['avg']) 
                d['sum'][1]+=int(p[1]) 
                d['sum'][2]+=int(p[2]) 
                d['sum'][3]+=int(p[3]) 
        print(d['sum'][1]/(len(d)-1),d['sum'][2]/(len(d)-1),d['sum'][3]/(len(d)-1)) 
testfile = open("test.txt", "r", encoding='utf-8') 
getmarks(testfile) 
testfile.close() 
"""
""" 
import math 
print(2*math.pi*float(input())) 
"""
""" 
import sys 
for i in range(1,len(sys.argv)): 
    print(sys.argv[i],end=' ') 
"""
""" 
n=int(input()) 
d={} 
for i in range(n): 
        x=input().split(';') 
        if x[0] in d: 
                d[x[0]]['cnt']+=1 
                d[x[0]]['w']+=int(int(x[1])>int(x[3])) 
                d[x[0]]['f']+=int(int(x[1])<int(x[3])) 
                d[x[0]]['o']+=int(int(x[1])==int(x[3])) 
        else: 
                d[x[0]]={'cnt':1,'w': int(int(x[1])>int(x[3])),'f': int(int(x[1])<int(x[3])),'o':int(int(x[1])==int(x[3]))} 
        if x[2] in d: 
                d[x[2]]['cnt']+=1 
                d[x[2]]['w']+=int(int(x[3])>int(x[1])) 
                d[x[2]]['f']+=int(int(x[3])<int(x[1])) 
                d[x[2]]['o']+=int(int(x[3])==int(x[1])) 
        else: 
                d[x[2]]={'cnt':1,'w': int(int(x[3])>int(x[1])),'f': int(int(x[3])<int(x[1])),'o':int(int(x[3])==int(x[1]))} 
for i in d: 
        print(i+':'+str(d[i]['cnt']),d[i]['w'],d[i]['o'],d[i]['f'],(d[i]['w']*3+d[i]['o'])) 
"""
""" 
s0=[i for i in input()] 
s1=[i for i in input()] 
d={} 
for i in range(len(s0)): 
        d[s0[i]]=s1[i] 
for i in input(): 
        print(d[i],end='') 
print() 
for i in input(): 
        for j in d: 
                if i==d[j]: 
                        print(j,end='') 
"""
""" 
n=int(input()) 
d=[] 
for i in range(n): 
        d.append(input().lower()) 
#t=[] 
e=[] 
n=int(input()) 
for i in range(n): 
        #t.append(input()) 
        for x in input().split(): 
                if (not x.lower() in d) and (not x.lower() in e): 
                        e.append(x.lower()) 
for i in e: 
        print(i) 
"""
""" 
n=int(input()) 
x=0 
y=0 
for i in range(n): 
        p=input().split() 
        if p[0]=='север': 
                y+=int(p[1]) 
        elif p[0]=='юг': 
                y-=int(p[1]) 
        elif p[0]=='восток': 
                x+=int(p[1]) 
        else: 
                x-=int(p[1]) 
print(x,y) 
"""
""" 
def getstat(l): 
        d={} 
        for line in l: 
                p=line.replace('\n', '').split() 
                #print(p) 
                if not int(p[0]) in d: 
                        d[int(p[0])]={'s':int(p[2]),'c':1} 
                else: 
                        d[int(p[0])]['s']+=int(p[2]) 
                        d[int(p[0])]['c']+=1 
        print(d) 
        for i in range(1,12): 
                if not i in d: 
                        print(i,'-') 
                else: 
                        print(i,d[i]['s']/d[i]['c']) 
testfile = open("test.txt", "r", encoding='utf-8') 
getstat(testfile) 
testfile.close() 
"""
""" 
Вашей программе будет доступна функция foo, которая может бросать исключения. 
Вам необходимо написать код, который запускает эту функцию, затем ловит исключения ArithmeticError, AssertionError, ZeroDivisionError и выводит имя пойманного исключения. 
Пример решения, которое вы должны отправить на проверку. 
try: 
    foo() 
except Exception: 
    print("Exception") 
except BaseException: 
    print("BaseException") 
Подсказка: https://docs.python.org/3/library/exceptions.html#exception-hierarchy 

try: 
    foo() 
except ZeroDivisionError: 
    print("ZeroDivisionError") 
except ArithmeticError: 
    print("ArithmeticError") 
except AssertionError: 
    print("AssertionError") 
"""
""" 
def getIsSubClass(c1,lst): 
        b=False 
        if c1 in lst: 
                return True 
        else: 
                #print(lst) 
                if len(s[c1]['pid'])==0: 
                        return False 
                else: 
                        for c2 in s[c1]['pid']: 
                                if c2 in lst: 
                                        return True 
                                else: 
                                        b=b or getIsSubClass(c2,lst) 
                        return b 
n=int(input()) 
s={} 
for i in range(n): 
        x=[i for i in input().split()] 
        if ":" in x: 
                s[x[0]]={'pid':x[2:]} 
                for k in x[2:]: 
                        if k not in s: 
                                s[k]={'pid':[]} 
        else: 
                s[x[0]]={'pid':[]} 
#print(s) 
q=int(input()) 
l=[] 
for i in range(q): 
        a=input() 
        if getIsSubClass(a,l): 
                print(a) 
        l.append(a) 
"""
""" 
try: 
    foo() 
except ZeroDivisionError: 
    print("ZeroDivisionError") 
except ArithmeticError: 
    print("ArithmeticError") 
except AssertionError: 
    print("AssertionError") 
"""
""" 
class NonPositiveError(Exception): 
        pass 
class PositiveList(list): 
        def append(self,x): 
                if x >= 0: 
                        raise NonPositiveError 
                else: 
                        self.append(x) 
"""
""" 
import datetime 
y,m,d=[int(i) for i in input().split()] 
dcnt=int(input()) 
date_=(datetime.date(y,m,d)+datetime.timedelta(days=dcnt)) 
#print((datetime.date(y,m,d)+datetime.timedelta(days=dcnt)).strftime("%Y %m %d")) 
print(date_.year,date_.month,date_.day) 
"""
""" 
from simplecrypt import encrypt, decrypt 

with open("d:\encrypted.bin", "rb") as inp: 
    encrypted = inp.read() 

testfile = open("d:\passwords.txt", "r", encoding='utf-8') 
for line in testfile: 
        try: 
                print(decrypt(line.replace('\n', '').strip(),encrypted)) 
        except Exception: 
                print("Exception") 
        finally: 
                pass 
testfile.close() 
"""
""" 
class multifilter: 
    def judge_half(pos, neg): 
        # допускает элемент, если его допускает хотя бы половина фукнций (pos >= neg) 
        return pos >= neg 
    def judge_any(pos, neg): 
        # допускает элемент, если его допускает хотя бы одна функция (pos >= 1) 
        return pos >= 1 
    def judge_all(pos, neg): 
        # допускает элемент, если его допускают все функции (neg == 0) 
        return neg == 0 
    def __init__(self, iterable, *funcs, judge=judge_any): 
        # iterable - исходная последовательность 
        # funcs - допускающие функции 
        # judge - решающая функция 
        self.iterable = iterable 
        self.judge = judge 
        self.funcs = funcs 
        #pos=len([x for x in iterable for f in funcs if f(x)]) 
    def __iter__(self): 
        # возвращает итератор по результирующей последовательности 
        for x in self.iterable: 
            pos=0 
            neg=0 
            for f in self.funcs: 
                if f(x): 
                    pos+=1 
                else: 
                    neg+=1 
            if self.judge(pos,neg): 
                yield x 
"""
""" 
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27] 
print(list(filter(lambda x: x % 3 == 0, foo))) 
[18, 9, 24, 12, 27] 


Целое положительное число называется простым, если оно имеет ровно два различных делителя, то есть делится только на единицу и на само себя. 
Например, число 2 является простым, так как делится только на 1 и 2. Также простыми являются, например, числа 3, 5, 31, и еще бесконечно много чисел. 
Число 4, например, не является простым, так как имеет три делителя – 1, 2, 4. Также простым не является число 1, так как оно имеет ровно один делитель – 1. 

Реализуйте функцию-генератор primes, которая будет генерировать простые числа в порядке возрастания, начиная с числа 2. 

Пример использования:﻿ 
print(list(itertools.takewhile(lambda x : x <= 31, primes()))) 
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] 

import math 
def is_prime(a): 
        if a<=1: 
                return False 
        if a%2==0: 
                return False 
        for i in range(3,int(math.sqrt(a))+1,1):         
                if a%i==0: 
                        return False 
        return True 
def primes(): 
    x=2 
    yield 2 
    while True: 
        x+=1 
        if is_prime(x): 
            yield x 
"""
""" 
d=[] 
with open("dataset_24465_4.txt") as f: 
        for l in f: 
                d.append(l.strip()) 
with open("new2.txt","w") as f: 
        for l in range(len(d)-1,-1,-1): 
                f.write(d[l]+'\n') 
"""
""" 
import os 
path="main" 
d=[] 
for (path,dirs,files) in os.walk(path): 
        if len(list(filter(lambda file : file[-3:]==".py", files)))>0: 
                if not path in d: 
                        d.append(path) 
with open("dirs.txt","w") as f: 
        for i in sorted(d): 
                #print(i) 
                f.write(i+'\n') 
"""
""" 
def mod_checker(x, mod=0): 
        return (lambda y: y%x==mod) 
mod_3 = mod_checker(3) 

print(mod_3(3)) # True 
print(mod_3(4)) # False 

mod_3_1 = mod_checker(3, 1) 
print(mod_3_1(4)) # True 
"""
""" 
s, a, b = [input().strip() for _ in range(3)] 
if a in s: 
        if a in b: 
                print("Impossible") 
        else: 
                cnt = 0 
                while a in s: 
                        cnt += 1 
                        s = s.replace(a,b) 
                print(cnt) 
else: 
        print(0) 
"""
""" 
s, t = [input().strip() for _ in range(2)] 
cnt = 0 
i = 0 
while i < len(s): 
        if s.find(t,i) > -1: 
                #print(s.find(t,i)) 
                if s.find(t,i) > i: 
                        i = s.find(t,i)+1 
                else: 
                        i += 1 
                cnt += 1 
        else: 
                i += 1 
print(cnt) 
"""
""" 
import sys 
import re 

for line in sys.stdin: 
        line = line.rstrip() 
        #p = re.compile('(cat(.*)){2,}') 
        #if not p.match(line) is None: 
        #        print(line) 
        p = re.compile('(cat(.*)){2,}') 
        #print(p.search(line)) 
        if not (p.search(line) is None): 
                print(line) 
"""
""" 
import sys 
import re 

for line in sys.stdin: 
        line = line.rstrip() 
        #p = re.compile('(cat(.*)){2,}') 
        #if not p.match(line) is None: 
        #        print(line) 
        p = re.compile(r'\bcat\b') 
        #print(p.search(line)) 
        if not (p.search(line) is None): 
                print(line) 
"""
""" 
import sys 
import re 

for line in sys.stdin: 
        line = line.rstrip() 
        if not (re.search(r'z(.{3})z',line) is None): 
                print(line) 
"""
""" 
import sys 
import re 

for line in sys.stdin: 
        line = line.rstrip() 
        if not (re.search(r'.*\\.*',line) is None): 
                print(line) 
"""
""" 
>>> p = re.compile(r'(\b\w+)\s+\1') 
>>> p.search('Paris in the the spring').group() 
'the the' 

import sys 
import re 

for line in sys.stdin: 
        line = line.rstrip() 
        if not (re.search(r'\b([a-zA-Z0-9]+)\1\b',line) is None): 
                print(line) 
"""
""" 
import sys 
import re 
for line in sys.stdin: 
        line = line.rstrip() 
        print(re.sub('(human)','computer',line)) 
"""
""" 
Вам дана последовательность строк. 
В каждой строке замените первое вхождение слова, состоящего только из латинских букв "a" (регистр не важен), на слово "argh". 
Примечание: 
Обратите внимание на параметр count у функции sub﻿. 
Sample Input: 
There’ll be no more "Aaaaaaaaaaaaaaa" 
AaAaAaA AaAaAaA 
Sample Output: 
There’ll be no more "argh" 
argh AaAaAaA 

import sys, re 
for line in sys.stdin: 
        line = line.rstrip() 
        print(re.sub(r'\b([aA]+)\b','argh',line, count = 1)) 
"""
""" 
import sys, re 
def repl(m): 
        return m.group(2)+m.group(1)+m.group(3) 

for line in sys.stdin: 
        line = line.rstrip() 
        print(re.sub(r'\b(\w)(\w)(\w*)\b',repl,line)) 
"""
""" 
buzzzzzbasdasdzzzzz -> buzbasdasdz 
import sys, re 
def repl(m): 
        return m.group(1) 

for line in sys.stdin: 
        line = line.rstrip() 
        print(re.sub(r'(\w)\1+',repl,line)) 
"""
""" 
Sample Input: 
0 
10010 
00101 
01001 
Not a number 
1 1 
0 0 

Sample Output: 
0 
10010 
01001 
"""
""" 
import sys, re 
def d3(t): 
        multiplier = 1 
        accumulator = 0 
        for bit in t: 
                accumulator = (accumulator + int(bit) * multiplier) % 3 
                multiplier = 3 - multiplier 
        return accumulator == 0 

#print(re.findall(r'\d','0101111')) 
for line in sys.stdin: 
        line = line.rstrip() 
        if not (re.search(r'^([01]+)\Z',line) is None): 
                #print(line) 
                if d3(line): 
                        print(line) 
        #print(re.sub(r'(\w)\1+',repl,line)) 
"""
""" 
import urllib 

link = "https://stepic.org/media/attachments/lesson/24472/sample0.html" 
try: 
        #f = urllib.urlopen(link) 
        #print(f) 
        #myfile = f.read() 
        #print(myfile) 
        
        req = urllib.request.Request(link) 
        with urllib.request.urlopen(req) as response: 
                the_page = response.read() 
        print(the_page) 
        
        #import requests 
        #f = requests.get(link) 
        #print(f.text) 
except Exception as e: 
        pass 
"""
""" 
import csv 
d={} 
with open('Crimes.csv') as csvfile: 
        lines = csv.DictReader(csvfile, delimiter=',') 
        for line in lines: 
                if not line['Primary Type'] in d: 
                        d[line['Primary Type']] = 0 
                else: 
                        d[line['Primary Type']] += 1 
#print(d) 
#for w in sorted(d, key=d.get, reverse=True): 
#        print(w, d[w]) 
print(sorted(d, key=d.get, reverse=True)[0]) 
"""
""" 
import json 

def getIsSubClass(c1,c2): 
        b=False 
        #if c1==c2: 
        #        return True 
        #else: 
        if len(dd[c2]['pid'])==0: 
                return False 
        else: 
                if c1 in dd[c2]['pid']: 
                        return True 
                else: 
                        for j in dd[c2]['pid']: 
                                b=b or getIsSubClass(c1,j) 
                        return b 

d=json.loads(input()) 
dd={} 
for i in d: 
        dd[i['name']]={'pid':i['parents'],'cnt':0} 
for i in dd: 
        for j in dd: 
                if getIsSubClass(i,j): 
                        dd[i]['cnt'] += 1 

for i in sorted(dd): 
        print(i,':',dd[i]['cnt']+1) 
"""
""" 
http://numbersapi.com/31/math?json=true 
 { 
 "text": "31 is a repdigit in base 5 (111), and base 2 (11111).", 
 "number": 31, 
 "found": true, 
 "type": "math" 
} 
http://numbersapi.com/999/math?json=true 
{ 
 "text": "999 is an unremarkable number.", 
 "number": 999, 
 "found": false, 
 "type": "math" 
} 
"""
""" 
import urllib.request 
import json 

#link = "http://numbersapi.com/31/math?json=true" 
try: 
        #req = urllib.request.Request(link) 
        #with urllib.request.urlopen(link) as response: 
        #        the_page = response.read() 
        num=int(input()) 
        with urllib.request.urlopen("http://numbersapi.com/"+num+"/math?json=true") as response: 
                the_page = response.read() 
        d=json.loads(the_page) 
        for i in d: 
                if i['found']: 
                        print('Interesting') 
                else: 
                        print('Boring') 
except Exception as e: 
        print(e) 
"""
""" 
import urllib.request, json 
url = "http://numbersapi.com/31/math?json=true" 
response = urllib.request.urlopen(url) 
data = json.loads(response.read()) 
print(data) 
"""
""" 
import requests 
import re 

def checkWayBy2Step(url_b, url_e): 
    res = requests.get(url_b) 
    if res.status_code==200: 
        #print(res.text) 
        for url2 in re.findall('\<a\s.*href=\"(.*)\">',res.text): 
            #print(url2) 
            res2 = requests.get(url2) 
            if res2.status_code==200: 
                for url3 in re.findall('\<a\s.*href=\"(.*)\">',res2.text): 
                    if url3 == url_e: 
                        return True 
    return False 

url1='https://stepic.org/media/attachments/lesson/24472/sample0.html'#input() 
url2='https://stepic.org/media/attachments/lesson/24472/sample0.html'#input() 
if checkWayBy2Step(url1,url2): 
    print('Yes') 
else: 
    print('No') 
"""
""" 
import requests 
import re 

def getUrl(url): 
        try: 
                if (re.search(r'^\.\.',url) is None): 
                        if not (re.search(r'\/\/',url) is None): 
                                return re.findall(r'^(([^:/?#]+):)?(//([^/?#:\"\']*))?([^?#]*)(\?([^#]*))?(#(.*))?',url)[0][3] 
                        else: 
                                return re.findall(r'^(([^:/?#]+):)?(//([^/?#:\"\']*))?([^?#]*)(\?([^#]*))?(#(.*))?',url)[0][4] 
        except Exception: 
                pass 
d = [] 
#ur='https://stepic.org/media/attachments/lesson/24471/03'#input() 
#res = requests.get(ur) 
#if res.status_code == 200: 
#        for url in re.findall('\<a\s.*href=[\"\'](.*)[\"\'\s]+?>',res.text): 
with open('urls.txt') as f: #, "r", "UTF-8" 
        for line in f: 
                #for url in re.findall('\<a\s.*href=[\"\'](.*)[\"\'\s]>',line): 
                #print(line) 
                for url in re.findall('\<a\s.*href=[\"\'](.*)[\"\'\s]>',line): 
                        u = getUrl(url) 
                        if not u is None: 
                                if not u in d: 
                                        d.append(u) 
for i in sorted(d): 
        print(i) 

#d[i] = {'url':getUrl(line)} 
#print(line) 
#with open('urls.txt') as f: 
#for line in f: 
"""
""" 
def url_path_to_dict(path): 
    pattern = (r'^' 
               r'((?P<schema>.+?)://)?' 
               r'((?P<user>.+?)(:(?P<password>.*?))?@)?' 
               r'(?P<host>.*?)' 
               r'(:(?P<port>\d+?))?' 
               r'(?P<path>/.*?)?' 
               r'(?P<query>[?].*?)?' 
               r'$' 
               ) 
    regex = re.compile(pattern) 
    m = regex.match(path) 
    d = m.groupdict() if m is not None else None 

    return d 

def main(): 
    print url_path_to_dict('http://example.example.com/example/example/example.html') 
"""

""" 
# correct 
import requests 
import re 

def url_path_to_dict(path): 
    pattern = (r'^' 
               r'((?P<schema>.+?)://)?' 
               r'((?P<user>.+?)(:(?P<password>.*?))?@)?' 
               r'(?P<host>.*?)' 
               r'(:(?P<port>\d+?))?' 
               r'(?P<path>/.*?)?' 
               r'(?P<query>[?].*?)?' 
               r'$' 
               ) 
    regex = re.compile(pattern) 
    m = regex.match(path) 
    d = m.groupdict() if m is not None else None 

    return d 

d = [] 
ur=input() 
res = requests.get(ur) 
if res.status_code == 200: 
    for url in re.findall('\<a\s.*href=[\"\'](.*?)[\"\'\s]+?',res.text): 
        if (re.search(r'^\.\.',url) is None): 
            u = url_path_to_dict(url) 
            if not u['host'] is None: 
                if not u['host'] in d: 
                    d.append(u['host']) 
else: 
        print(res.status_code) 
for i in sorted(d): 
        print(i) 
"""
""" 
В этой задаче вам необходимо воспользоваться API сайта numbersapi.com 
Вам дается набор чисел. Для каждого из чисел необходимо узнать, существует ли интересный математический факт об этом числе. 
Для каждого числа выведите Interesting, если для числа существует интересный факт, и Boring иначе. 
Выводите информацию об интересности чисел в таком же порядке, в каком следуют числа во входном файле. 
Пример запроса к интересному числу: 
http://numbersapi.com/31/math?json=true 
Пример запроса к скучному числу: 
http://numbersapi.com/999/math?json=true 
Пример входного файла: 
31 
999 
1024 
502 
﻿Пример выходного файла: 
Interesting 
Boring 
Interesting 
Boring 
#--------------------------------------------------- 
import requests 
import re 
import json 

try: 
        with open("d:\\_Projects\\_OLD\\input.txt") as f: 
                for line in f: 
                        num=line.strip() 
                        res = requests.get("http://numbersapi.com/"+num+"/math?json=true") 
                        if res.status_code == 200: 
                                d=json.loads(res.text) 
                                with open("d:\\_Projects\\_OLD\\output.txt","a") as ff: 
                                        if d['found']: 
                                                #print('Interesting') 
                                                ff.write('Interesting'+'\n') 
                                        else: 
                                                #print('Boring') 
                                                ff.write('Boring'+'\n') 
except Exception as e: 
        print(e) 
"""
""" 
В этой задаче вам необходимо воспользоваться API сайта artsy.net 
API проекта Artsy предоставляет информацию о некоторых деятелях искусства, их работах, выставках. 
В рамках данной задачи вам понадобятся сведения о деятелях искусства (назовем их, условно, художники). 
Вам даны идентификаторы художников в базе Artsy. 
Для каждого идентификатора получите информацию о имени художника и годе рождения. 
Выведите имена художников в порядке неубывания года рождения. В случае если у художников одинаковый год рождения, выведите их имена в лексикографическом порядке. 
Работа с API Artsy 
Полностью открытое и свободное API предоставляют совсем немногие проекты. В большинстве случаев, для получения доступа к API необходимо зарегистрироваться в проекте, создать свое приложение, и получить уникальный ключ (или токен), и в дальнейшем все запросы к API осуществляются при помощи этого ключа. 
Чтобы начать работу с API проекта Artsy, вам необходимо пройти на стартовую страницу документации к API https://developers.artsy.net/start и выполнить необходимые шаги, а именно зарегистрироваться, создать приложение, и получить пару идентификаторов Client Id и Client Secret. Не публикуйте эти идентификаторы. 
После этого необходимо получить токен доступа к API. На стартовой странице документации есть примеры того, как можно выполнить запрос и как выглядит ответ сервера. Мы приведем пример запроса на Python. 
import requests 
import json 
client_id = '...' 
client_secret = '...' 
# инициируем запрос на получение токена 
r = requests.post("https://api.artsy.net/api/tokens/xapp_token", 
                  data={ 
                      "client_id": client_id, 
                      "client_secret": client_secret 
                  }) 
# разбираем ответ сервера 
j = json.loads(r.text) 
# достаем токен 
token = j["token"] 
Теперь все готово для получения информации о художниках. На стартовой странице документации есть пример того, как осуществляется запрос и как выглядит ответ сервера. Пример запроса на Python. 
# создаем заголовок, содержащий наш токен 
headers = {"X-Xapp-Token" : token} 
# инициируем запрос с заголовком 
r = requests.get("https://api.artsy.net/api/artists/4d8b92b34eb68a1b2c0003f4", headers=headers) 
# разбираем ответ сервера 
j = json.loads(r.text) 
Обратите внимание, что для сортировки художников по имени используется параметр sortable_name. 
Пример входных данных: 
4d8b92b34eb68a1b2c0003f4 
537def3c139b21353f0006a6 
4e2ed576477cc70001006f99 
Пример выходных данных: 
Abbott Mary 
Warhol Andy 
Abbas Hamra 
﻿Примечание для пользователей Windows 
При открытии файла для записи на Windows ﻿по ﻿умолчанию ﻿используется кодировка CP1251, в то время как для записи имен на сайте используется кодировка UTF-8, что может привести к ошибке при попытке записать в файл имя с необычными символами. 


import requests 
import json 

client_id = '4cf6100d0975c9763481' 
client_secret = 'd05320f6d690dfabaee6ff7dd7809470' 
r = requests.post("https://api.artsy.net/api/tokens/xapp_token", 
                  data={ 
                      "client_id": client_id, 
                      "client_secret": client_secret 
                  }) 
j = json.loads(r.text) 
token = j["token"] 
headers = {"X-Xapp-Token" : token} 
artists={} 
with open("d:\\_Projects\\_OLD\\input.txt") as f: 
        for line in f: 
                num = line.strip() 
                r = requests.get("https://api.artsy.net/api/artists/"+num, headers=headers) 
                q = json.loads(r.text) 
                #print(q) 
                artists[q['sortable_name'].encode('utf8')]=q['birthday'] 
print(sorted(artists.items(), key=lambda x: (x[1], x[0]))) 
with open("d:\\_Projects\\_OLD\\output.txt","wb") as ff: 
        for i in [v[0] for v in sorted(artists.items(), key=lambda x: (x[1], x[0]))]: 
                ff.write(i+b'\n')#.decode('utf8') 
"""

""" 
Вам дано описание пирамиды из кубиков в формате XML. 
Кубики могут быть трех цветов: красный (red), зеленый (green) и синий (blue﻿). 
Для каждого кубика известны его цвет, и известны кубики, расположенные прямо под ним. 
Пример: 
<cube color="blue"> 
  <cube color="red"> 
    <cube color="green"> 
    </cube> 
  </cube> 
  <cube color="red"> 
  </cube> 
</cube> 
Введем понятие ценности для кубиков. Самый верхний кубик, соответствующий корню XML документа имеет ценность 1. Кубики, расположенные прямо под ним, имеют ценность 2. Кубики, расположенные прямо под нижележащими кубиками, имеют ценность 3. И т. д. 
Ценность цвета равна сумме ценностей всех кубиков этого цвета. 
Выведите через пробел три числа: ценности красного, зеленого и синего цветов. 
Sample Input: 
<cube color="blue"><cube color="red"><cube color="green"></cube></cube><cube color="red"></cube></cube> 
Sample Output: 
4 3 1 

import xml.etree.ElementTree as etree 
def gatherdict(r,lev): 
        if not r.attrib['color'] in d: 
                d[r.attrib['color']] = lev 
        else: 
                d[r.attrib['color']] += lev 
        for c in r: 
                if len(c.attrib['color'])>0: 
                        gatherdict(c, lev + 1) 
d={} 
n=input() 
root = etree.fromstring(n) 
gatherdict(root,1) 
s = '' 
if 'red' in d: 
    s = str(d['red']) 
else: 
    s = '0' 
if 'green' in d: 
    s += ' ' + str(d['green']) 
else: 
    s += ' 0' 
if 'blue' in d: 
    s += ' ' + str(d['blue']) 
else: 
    s += ' 0' 
print(s)
"""

k, n = [int(i) for i in input().split()]
# print(k,n)
if k >= n:
    print(0)
else:
    for k1 in range(k):
        for i in range(n):
            p = [i]
            for j in range(i + k1 + 1, n):
                # if i ==0 and j == 3:
                #	print(i,j,k1,i+k1+1,p)
                if len(p) < k:
                    p.append(j)
            # else:
            #	print(p)
            #	p=[i,j]
            if len(p) == k:
                print(p)

pool = tuple(iterable)
n = len(pool)
r = n if r is None else r
if r > n:
    return
indices = range(n)
cycles = range(n, n - r, -1)
yield tuple(pool[i] for i in indices[:r])
while n:
    for i in reversed(range(r)):
        cycles[i] -= 1
        if cycles[i] == 0:
            indices[i:] = indices[i + 1:] + indices[i:i + 1]
            cycles[i] = n - i
        else:
            j = cycles[i]
            indices[i], indices[-j] = indices[-j], indices[i]
            yield tuple(pool[i] for i in indices[:r])
            break
    else:
        return

import itertools

k, n = [int(i) for i in input().split()]
for p in itertools.combinations(range(n), k):
    print(' '.join(str(x) for x in p))


def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


"""

Лаборатория
Ввести с клавиатуры строку латиницей (1-3 слова). Зашифровать ее с использованием гарантированных алгоритмов шифрования. Сформировать словарь, где в качестве ключа используется название гарантированного алгоритма шифрования, а в качестве значения - результат шифрования в шестнадцатеричном представлении { 'sha1': 'aaf4c…', 'md5', '5d4…',…}.
Итог вывести отдельными операторами вывода в виде пар ключа и значения, отсортированных по возрастанию ключа:
md5 5d414…
sha1 aaf4c…
"""
import hashlib

s = input()
h = {}
h['md5'] = (hashlib.md5(s.encode())).hexdigest()
h['sha1'] = (hashlib.sha1(s.encode())).hexdigest()
h['sha224'] = (hashlib.sha224(s.encode())).hexdigest()
h['sha256'] = (hashlib.sha256(s.encode())).hexdigest()
h['sha384'] = (hashlib.sha384(s.encode())).hexdigest()
h['sha512'] = (hashlib.sha512(s.encode())).hexdigest()
for i in sorted(h):
    print(i, h[i])

###########################
###########################
"""
2.2 Числа Фибоначчи
Задача на программирование: небольшое число Фибоначчи
Дано целое число 1≤n≤401≤n≤40, необходимо вычислить nn-е число Фибоначчи (напомним, что F0=0F0=0, F1=1F1=1 и Fn=Fn−1+Fn−2Fn=Fn−1+Fn−2 при n≥2n≥2).
Sample Input:
3
Sample Output:
2

def fib(n):
    if n==0:
        return 0
    if n==1:
        return 1
    x=[]
    x.append(0)
    x.append(1)
    for i in range(2,n+1):
       x.append(x[i-1]+x[i-2])
    return x[n]

def main():
    n = int(input())
    print(fib(n))

if __name__ == "__main__":
    main()

------------ alternate
def fib(n):
	a,b=0,1
	yield a
	yield b
	for i in range(n-1):
		a,b=b,a+b
		yield b

print(list(fib(int(input())))[-1])


-------- alternate
def fib(num):

    prev, cur = 0, 1

    for i in range(1, num):
        prev, cur = cur, prev + cur

    return cur


def main():
    n = int(input())
    print(fib(n))

if __name__ == "__main__":
    main()

"""
"""
Задача на программирование: последняя цифра большого числа Фибоначчи



Дано число 1≤n≤1071≤n≤107, необходимо найти последнюю цифру nn-го числа Фибоначчи.

Как мы помним, числа Фибоначчи растут очень быстро, поэтому при их вычислении нужно быть аккуратным с переполнением. 
В данной задаче, впрочем, этой проблемы можно избежать, поскольку нас интересует только последняя цифра числа Фибоначчи: 
если 0≤a,b≤9 — последние цифры чисел Fi и Fi+1 соответственно, 
то (a+b)mod10 — последняя цифра числа Fi+2.

Sample Input:
132941
Sample Output:
1

def fib_digit(n):
    if n==0:
        return 0
    if n==1:
        return 1
    x=[]
    x.append(0)
    x.append(1)
    for i in range(2,n+1):
       x.append((x[i-1]%10+x[i-2]%10)%10)
    return x[n]

def main():
    n = int(input())
    print(fib_digit(n))


if __name__ == "__main__":
    main()

--------- alternate!
def fibd(n):
	a,b=0,1
	for i in range(n-1):
		a,b=b%10,(a+b)%10
	return b
		
print(fibd(int(input())))	


----------- alternate
def fib_digit(n):
    d = [0,1]
    [d.append((d.pop(0) + d[-1])%10) for i in range(2,n+1)]
    return d[-1]


def main():
    n = int(input())
    print(fib_digit(n))


if __name__ == "__main__":
    main()

	
	
	
------- java alternate
import java.util.Scanner;
class Main {
  public static void main(String[] args) {
      Scanner sc = new Scanner(System.in);
      int i = Integer.parseInt(sc.nextLine()) + 1;
      int[] f = new int[i];
      f[0] = 0;
      f[1] = 1;
      for(int j = 2; j < f.length; j++)
      {
          f[j] = (f[j-1] + f[j-2]) %10;
      }
      System.out.print(f[i-1]);
  }
}
"""

"""
Задача на программирование повышенной сложности: огромное число Фибоначчи по модулю

Даны целые числа 1≤n≤10181≤n≤1018 и 2≤m≤1052≤m≤105, необходимо найти остаток от деления nn-го числа Фибоначчи на mm.

Sample Input:
10 2
Sample Output:
1

def fib_mod(n, m):
    x=[]
    x.append(0)
    x.append(1)
    for i in range(2,m*6):
	    x.append((x[i-1]+x[i-2])%m)
	    if (x[i]==1) & (x[i-1]==0): #this is a period!
	        break
    return x[(n%(len(x)-2))]

def main():
    n, m = map(int, input().split())
    print(fib_mod(n, m))


if __name__ == "__main__":
    main()

------------- alternate
n,m=map(int,input().split())
o,i=[0,1],2
while not (o[i-2]==0 and o[i-1]==1) or i<=2:
	o.append((o[i-2]+o[i-1])%m)
	i+=1
print(o[n%(i-2)])

-------------- alternate
n, MOD = map(int, input().split())

def mul(A,B):
    return [ [ sum(A[r][i]*B[i][c] for i in range(2))%MOD for c in range(2) ] for r in range(2) ]

def exp(A,n):
    if n==0: return [ [1,0], [0,1] ]
    C = exp(A,n//2)
    C = mul(C,C)
    return mul(A,C) if n%2 else C

print( exp( [ [0,1], [1,1] ], n )[0][1] ) 
http://codeforces.com/blog/entry/14516?locale=ru	
	
"""

"""
Задача на программирование: наибольший общий делитель
По данным двум числам 1≤a,b≤2⋅1091≤a,b≤2⋅109 найдите их наибольший общий делитель.
Sample Input 1:
18 35
Sample Output 1:
1
Sample Input 2:
14159572 63967072
Sample Output 2:
4

def gcd(a, b):
	if a==0:
		return b
	if b==0:
		return a
	if a>b:
		return gcd(a%b,b)
	else:
		return gcd(a,b%a)
def main():
    a, b = map(int, input().split())
    print(gcd(a, b))
if __name__ == "__main__":
    main()
---------- alternate
x, y = sorted(map(int, input().split()))
while x:    
    y,  x = x, y % x
print(y)
------------ alternate
def gcd(a, b):
    while a:
        a, b = b % a, a
    return b
def main():
    a, b = map(int, input().split())
    print(gcd(a, b))
if __name__ == "__main__":
    main()
----------- alternate
def gcd(a, b):
    return gcd(b, a % b) if b else a    
def main():
    a, b = map(int, input().split())
    print(gcd(a, b))
if __name__ == "__main__":
    main()
-------------- 
"""

"""
Задача на программирование: покрыть отрезки точками
По данным nn отрезкам необходимо найти множество точек минимального размера, для которого каждый из отрезков содержит хотя бы одну из точек.
В первой строке дано число 1≤n≤1001≤n≤100 отрезков. Каждая из последующих nn строк содержит по два числа 0≤l≤r≤1090≤l≤r≤109, задающих начало и конец отрезка. 
Выведите оптимальное число mm точек и сами mm точек. Если таких множеств точек несколько, выведите любое из них.

Sample Input 1:
3
1 3
2 5
3 6
Sample Output 1:
1
3 
Sample Input 2:
4
4 7
1 3
2 5
5 6
Sample Output 2:
2
3 6 


n = int(input())
x = [[int(x) for x in input().split()] for i in range(n)]
x.sort(key=lambda t: t[1])
m=[]
for i in x:
	f=0
	for k in m:
		if k >= i[0] and k <= i[1]:
			f=1
	if f==0:
		m.append(i[1])
print(len(m))
print(' '.join(str(k) for k in m))

----------------- alternate
segments = sorted([sorted(map(int,input().split())) for i in range(int(input()))], key=lambda x: x[1])
dots = [segments.pop(0)[1]]
for l, r in segments:
    if l > dots[-1]:
        dots.append(r)
print(str(len(dots)) + '\n' + ' '.join(map(str, dots)))

-------------- alternate
def greedy(lines):
    lines = sorted(lines, key = lambda x: x[1])
    l, r = lines.pop(0)
    dots = [r]
    for l, r in lines:
        if l <= dots[-1] <= r:
            continue
        else:
            dots.append(r)
    dots = list(map(str, dots))
    return str(len(dots)) + '\n' + ' '.join(dots)

def main():
    n = int(input())
    lines = []
    for i in range(n):
        line = list(map(int, input().split()))
        lines.append(line)
    print(greedy(lines))


if __name__ == "__main__":
    main()
-------------- alternate
from operator import itemgetter

segments = [tuple(map(int, input().split())) for _ in range(int(input()))]
segments.sort(key=itemgetter(1))

points = []
i = 0
while i < len(segments):
    cur = segments[i]
    points.append(cur[1])
    j = i + 1
    while j < len(segments) and cur[1] >= segments[j][0]:
        i = j
        j += 1
    i = j

print(len(points))
print(' '.join(map(str, points)))

"""

"""
Задача на программирование: непрерывный рюкзак



Первая строка содержит количество предметов 1≤n≤1031≤n≤103 и вместимость рюкзака 0≤W≤2⋅106. 
Каждая из следующих nn строк задаёт стоимость 0≤ci≤2⋅1060≤ci≤2⋅106 и объём 0<wi≤2⋅1060<wi≤2⋅106 предмета (nn, WW, cici, wiwi — целые числа). 
Выведите максимальную стоимость частей предметов (от каждого предмета можно отделить любую часть, стоимость и объём при этом пропорционально уменьшатся), 
помещающихся в данный рюкзак, с точностью не менее трёх знаков после запятой.

Sample Input:
3 50
60 20
100 50
120 30
Sample Output:
180.000

n,max = [int(x) for x in input().split()]
x = [[int(x) for x in input().split()] for i in range(n)]
x.sort(key=lambda t: t[0]/t[1], reverse=True)
s=0
while max and x:
	s+=x[0][0] if x[0][1]<=max else x[0][0]*max/x[0][1]
	if x[0][1]<max:
		max=max-x[0][1]
		x.pop(0)
	else:
		max=0
print(s)


------------- alternate
# put your python code here
n, W = map(int, input('').split())
l = sorted([tuple(map(int, input('').split())) for N in range(n)], key=lambda x: x[0]/x[1], reverse=True)
ans = 0.0
for i in l:
    if W == 0:
        break
    if i[1] <= W:
        ans += i[0]
        W -= i[1]
    else:
        ans += (i[0]/i[1]) * W
        W = 0
print('{0:.3f}'.format(ans))


----------------- alternate
n, W = map(int, input().split())
p = []
for i in range(n):
    c, w = map(int, input().split())
    if c * w > 0 : p.append([c ,w])
p.sort(key=lambda cw: cw[1]/cw[0] )
ans = []
weight_in_pack = 0
while weight_in_pack < W and len(p):
    i = p.pop(0)
    ans.append(i if i[1]<=(W - weight_in_pack) else [i[0]/i[1]*(W - weight_in_pack), (W - weight_in_pack)])
    weight_in_pack = sum([a[1] for a in ans])
print(sum([a[0] for a in ans]))
    
----------- alternate
def greedy(items, W):
    items = sorted(items, key = lambda x: x[0]/x[1], reverse = True)
    res = 0
    for c, w in items:
        if w <= W:
            res += c
            W -= w
        else:
            res += c * (W / w)
            break
    return '%.3f' %res

def main():
    n, W = list(map(int, input().split()))
    items = []
    for i in range(n):
        item = list(map(int, input().split()))
        items.append(item)
    print(greedy(items, W))


if __name__ == "__main__":
    main()

-------------------- alternate
N, space = map(int, input().split())
items = [tuple(map(int, input().split())) for _ in range(N)]
items.sort(key=lambda x: x[0] / x[1], reverse=True)

total = 0
for item in items:
    if space >= item[1]:
        total += item[0]
        space -= item[1]
    else:
        total += space * (item[0] / item[1])
        break
        
print("{:.3f}".format(total))

"""

"""
Задача на программирование: различные слагаемые

По данному числу 1≤n≤1091≤n≤109 найдите максимальное число kk, для которого nn можно представить как сумму kk различных натуральных слагаемых. 
Выведите в первой строке число kk, во второй — kk слагаемых.

Sample Input 1:
4
Sample Output 1:
2
1 3 
Sample Input 2:
6
Sample Output 2:
3
1 2 3 
----- time limit
n = int(input())
x = []
i = 1
while n:
	if ((not (i in x)) and (not ((n-i) in x))):
		x.append(i)
		n-=i
	i+=1
print(len(x))
print(' '.join(str(i) for i in x))
--- good

n = int(input())
x = []
i = 1
while n:
	if n>=i:
		x.append(i)
		n-=i
		i+=1
	else:
		x[-1]+=n
		break
print(len(x))
print(*x, sep=' ')

--- good
import math

n = int(input())
if n <= 2:
	print(1)
	print(n)
else:
	x = []
	k=math.trunc((-1+math.sqrt(1+8*n))/2)
	print(k)
	for i in range(1,k):
		x.append(i)
	x.append(n-math.trunc(k*(k-1)/2))
	print(' '.join(str(i) for i in x))

------------- alternate
n = int(input(''))
ans = [0]
i = 0
while n >= 0:
    i += 1
    if i >= (n -i):
        ans.append(n)
        ans[0] += 1
        break
    n -= i    
    ans.append(i)
    ans[0] += 1
print(ans[0], '\n', ' '.join([str(x) for x in ans[1:]]))
-------------- alternate
n = int(input())
s = set()
k = 0
nt = 0

while n not in s:
    nt += 1
    if n - nt > nt:
        s.add(nt)
        n -= nt
    else:
        s.add(n)

print(len(s), '\n', ' '.join([str(x) for x in sorted(list(s))]))

---------- alternate
def calc_n_from_S(S):
    spisok = []
    from math import sqrt
    d_sqrt = int(sqrt(1+8*S))
    n = (-1+d_sqrt)//2
    summa = int(n/2*(n+1))
    raznost = S-summa
    if raznost == 0:
        spisok = [i for i in range(1, n+1)]
    elif raznost > n:
        spisok = [i for i in range(1, n+1)] + [raznost]
    elif raznost > 0:
        spisok = [i for i in range(1, n)] + [n+raznost]
    return spisok

S_n = int(input())
k_list = calc_n_from_S(S_n)
print(len(k_list))
print(*k_list, sep=' ')
----------- alternate
print(*k_list)
------------- alternate
n = int(input())
i = 1
numbers = []
while n > 2*i:
    n -= i
    numbers.append(i)
    i += 1

numbers.append(n)
print(i)
print(*numbers, sep=' ')
------------ alternate
n = int(input())
i = 1
numbers = []
while n > 0:
    if n - i >= 0:
        n -= i
        numbers.append(i)
        i += 1
    else:
        numbers[-1] += n
        break

print(len(numbers))
print(*numbers, sep=' ')


"""

"""
Теоретическая задача для самостоятельной проверки: сдача минимальным количеством монет

Постройте жадный алгоритм, который получает на вход натуральное число nn и за время O(n)O(n) находит минимальное число монет номиналом 
1 копейка, 5 копеек, 10 копеек и 25 копеек, с помощью которых можно выдать сдачу в nn копеек. 
(Как всегда, нужно описать алгоритм, доказать его корректность и оценку на время работы. 
Приводить псевдокод нужно только в том случае, если вам кажется, что он поможет читателю лучше понять ваш алгоритм.)

Приведите пример номиналов монет, для которых жадный алгоритм построит неоптимальное решение. 
В множество номиналов должна входить монета номиналом 1 копейка, чтобы любую сумму nn можно было разменять этими монетами.

У меня получился алгоритм надежным шагом которого является деление числа n на максимальный номинал, 
взятие целой части от получившегося значения(кол-во монет данного номинала) и переход к следующему номиналу меньшего
 достоинства с n = n - int(n/i)*i, где i max номинал. То есть один проход по списку номиналов до момента пока n не равно 0.

Время работы этого алгоритма, как мне кажется, будет меньше чем O(n).  Придумать пример с неоптимальным решением у меня не получилось.

P.S. Если я правильно понял задачу) Видимо это тривиальная задача раз тут нет ни одного комментария )
"""

"""
Задача на программирование: кодирование Хаффмана
По данной непустой строке ss длины не более 104104, состоящей из строчных букв латинского алфавита, постройте оптимальный беспрефиксный код. 
В первой строке выведите количество различных букв kk, встречающихся в строке, и размер получившейся закодированной строки. 
В следующих kk строках запишите коды букв в формате "letter: code". В последней строке выведите закодированную строку.

Sample Input 1:
a

Sample Output 1:
1 1
a: 0
0
Sample Input 2:
abacabad

Sample Output 2:
4 14
a: 0
b: 10
c: 110
d: 111
01001100100111
w = input()
f={}
for i in w:
	if i in f:
		f[i]+=1
	else:
		f[i]=1
#print(f)
d=sorted(f.items(), key=lambda x: (x[1]))
#print(d)
n=len(d)
h=[] #(symbol,freq,ordernum?,flag?) =>? (symbol,freq,id,pid,str)
def insert(a,s,i,pid,str):
	h.append((a,s,i,pid,str))
for i in range(1,n+1,1):
	insert(d[i-1][0],d[i-1][1],i-1,-1,'')
#print(h)
def extractMin():
	i=0
	min=999999
	for j in range(len(h)):
		if h[j][1]<min and h[j][3]==-1:
			min = h[j][1]
			i = j
	return(i,h[i][2],min,h[i][4])
for k in range(n,2*n-1,1):
	i,id,q,s = extractMin()
	h[i]=(h[i][0],q,id,k,s+'0')
	j,id,l,s = extractMin()
	h[j]=(h[j][0],l,id,k,s+'1')
	insert('',q+l,k,-1,'')
if len(h)==1:
	h[0]=(h[0][0],h[0][1],h[0][2],h[0][3],'0')
#print(h)
def gatherCode(symb,id,s):
	print(symb,id,s)
	for i in range(len(h)):
		if h[i][3]==id:
			if h[i][0]==symb:
				return s+(h[i][4] if h[i][4]!='' else '0')
			else:
				if h[i][0]=='':
					if id!=-1:
						return gatherCode(symb,h[i][2],s + h[i][4])
					else:
						return gatherCode(symb,h[i][2],s)
def gatherCode2(symb):
	s=''
	for i in range(len(h)):
		if h[i][0]==symb:
			id=h[i][2]
	while id!=-1:
		for i in range(len(h)):
			if h[i][2]==id:
				s=h[i][4]+s
				id=h[i][3]
				break
	return s
l=''
for i in w:
	l+=gatherCode2(i)#gatherCode(i,-1,'')
print(len(f),len(l))
for i in f:
	print(i+':',gatherCode2(i))#gatherCode(i,-1,'')
for i in w:
	print(gatherCode2(i),end='')#gatherCode(i,-1,'')
	
------- alternate
# put your python code here
from collections import Counter, OrderedDict

s = input()
c = Counter(s)
cc = list(c)
free = []
d1 = c.most_common()


while d1:
    if len(d1) > 1:
        m1 = d1.pop()
        m2 = d1.pop()
        free.append((m1[0], m1[0] + m2[0]))
        free.append((m2[0], m1[0] + m2[0]))
        c = Counter(dict(d1)) + Counter({m1[0] + m2[0]: m1[1] + m2[1]})
        d1 = c.most_common()
    else:
        m1 = d1.pop()
        free.append((m1[0], None))

d = dict()
ft = [x[0] for x in free]

for x in cc:
    i = ft.index(x)
    d[x] = ''
    while free[i][1] is not None:
        d[x] += str(i % 2)
        i = ft.index(free[i][1])
    if not d[x]:
        d[x] = '0'        

for x in d:
    d[x] = d[x][-1::-1]
ou = []
for x in s:
    ou.append(d[x])
u = ''.join(ou)
print(len(d), len(u))
for x in d:
    print('{}: {}'.format(x, d[x]))
print(u)


---- alternate
def get_tree(string):
    from heapq import heappop, heappush
    ss = set(string)
    heap = []
    for c in ss:
        heappush(heap, (string.count(c), c, None, None))
    while len(heap)>1:
        root_1 = heappop(heap)
        root_2 = heappop(heap)
        heappush(heap, (root_1[0]+root_2[0], "\0", root_1, root_2))
    return heap
def get_table(tupl, code, dict_):
    ch = tupl[1]
    left = tupl[2]
    right = tupl[3]
    if left:
        get_table(left, code+"1", dict_)
    if right:
        get_table(right, code+"0", dict_)
    if ch!="\0":
        dict_[ch] = code
def get_coded_string(s, slovar):
    ret = ''
    for c in s:
        ret += slovar.get(c, '')
    return ret

s = input()
if len(set(s))==1:
    d = {s[0]: '0'}
else:
    h = get_tree(s)
    d = {}
    get_table(h[0], "", d)
slovo = get_coded_string(s, d)
print(len(d), len(slovo))
for key, value in d.items():
    print(key, value, sep=": ")
print(slovo)

---------- alternate

class Node:
    def __init__(self, name, left_child=None, right_child=None, parent=None, weight=0):
        self.name = name
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent
        self.weight = weight
        
    def way(self):
        parent = self.parent
        lst = []
        while parent != None: 
            lst.append(parent)
            parent = parent.parent
        return lst
        
def sortkey(key):
    return tree[key][0]

s = input()
if len(s) ==1:
    print(1, 1)
    print(s,': ',0,sep='')
    print(0)
else:
    alphabet = {}
    for letter in s:
        if letter in alphabet:
            alphabet[letter] += 1
        else:
            alphabet[letter] = 1
    if len(alphabet) == 1:
        print(1, len(s))
        print(s[0],': ',0,sep='')
        for t in range (len(s)):
            print(0,sep='',end='')
    else:
        tree = {}
        table = {}
        i = 1
        for key in alphabet:
            tree[i] = [alphabet[key]]
            table[key] = i
            i += 1
        heap = list(tree.keys())
        heap.sort(key=sortkey)
        while len(heap) > 1:
            item0 = heap.pop(0)
            item1 = heap.pop(0)
            tree[i] = [tree[item0][0] + tree[item1][0], item0, item1]
            heap.append(i) 
            heap.sort(key=sortkey)
            i += 1
        for key in tree:
            tree[key].pop(0)
        tree2={}
        for leaf in tree:
            if len(tree[leaf]) == 0:
                tree2[leaf] = Node(name=leaf)
            else:
                tree2[leaf] = Node(name=leaf,left_child=tree[leaf][0],right_child=tree[leaf][1])
            tree2[leaf].parent = [key for key in tree if leaf in tree[key]]
        for key in tree2:
            try:
                if tree2[key].parent == []:
                    tree2[key].parent = None
                else:
                    tree2[key].parent = tree2[tree2[key].parent[0]]
            except IndexError:
                pass
        for key in tree2:
            try:
                tree2[key].left_child = tree2[tree2[key].left_child]
                tree2[key].right_child = tree2[tree2[key].right_child]
            except KeyError:
                pass
        code = {}
        for letter in table:
            slf = tree2[table[letter]]
            wayy = slf.way()
            wayy.reverse()
            wayy.append(slf)
            u = ''
            for i in range(len(wayy)-1):
                if wayy[i+1] == wayy[i].left_child:
                    u += '0'
                elif wayy[i+1] == wayy[i].right_child:
                    u += '1'
            code.update({letter:u})
        s1 = ''
        for char in s:
            s1 += code[char]
        print(len(code), len(s1))
        for key in sorted(code):
            print(key,': ',code[key],sep='')
        print(s1)
------ alternate
from collections import Counter
import heapq
from operator import itemgetter

st = input()
heap = []
c = Counter(st)
frequencies = c.most_common()
for f in frequencies:
    v = (f[1], f[0])
    heapq.heappush(heap, v)

tree = {}
while len(heap) > 1:
    v1 = heapq.heappop(heap)
    v2 = heapq.heappop(heap)
    priority = v1[0] + v2[0]
    value = v1[1] + v2[1]
    heapq.heappush(heap, (priority, value))
    tree[value] = (v1[1], v2[1])


root = heapq.heappop(heap)[1]
q = [(root, '')]
if len(tree) == 0:
    tree[root] = None
    q = [(root, '0')]

code_table = {}
while len(q):
    next_key, prefix_code = q.pop()
    next = tree.get(next_key, None)
    if next:
        q.append((next[0], prefix_code + '0'))
        q.append((next[1], prefix_code + '1'))
    else:
        code_table[next_key] = prefix_code

encoded = ''
for ch in st:
    encoded += code_table[ch]

print(len(code_table), len(encoded))
for ch, code in sorted(list(code_table.items()),key=itemgetter(0)):
    print(ch + ':', code)

print(encoded)


"""

"""
Задача на программирование: декодирование Хаффмана
Восстановите строку по её коду и беспрефиксному коду символов. 
В первой строке входного файла заданы два целых числа kk и ll через пробел — количество различных букв, встречающихся в строке, 
и размер получившейся закодированной строки, соответственно. 
В следующих kk строках записаны коды букв в формате "letter: code". Ни один код не является префиксом другого. 
Буквы могут быть перечислены в любом порядке. В качестве букв могут встречаться лишь строчные буквы латинского алфавита; каждая из этих букв встречается в строке хотя бы один раз. 
Наконец, в последней строке записана закодированная строка. Исходная строка и коды всех букв непусты. Заданный код таков, что закодированная строка имеет минимальный возможный размер.
В первой строке выходного файла выведите строку ss. Она должна состоять из строчных букв латинского алфавита. 
Гарантируется, что длина правильного ответа не превосходит 104104 символов.

Sample Input 1:
1 1
a: 0
0
Sample Output 1:
a
Sample Input 2:
4 14
a: 0
b: 10
c: 110
d: 111
01001100100111
Sample Output 2:
abacabad


n,m = map(int, input().split())
d={}
for i in range(n):
	l,c = input().split()
	d[c]=l[0]
#print(d)
w=input()
wd=''
while len(w)>0:
	for x in d:
		if w.startswith(x):
			wd+=d[x]
			w=w[len(x):]
			break
print(wd)


--------- alternate
# put your python code here
from heapq import *

a = input().split(' ')
k, l = int(a[0]), int(a[1])

freq = {}

for _ in range(k):
    a = input().split(': ')
    freq[a[1]] = a[0]

enc_str = input()
dec_str = ""
cur_code = ""
for ch in enc_str:
    cur_code += ch
    if freq.get(cur_code):
        dec_str += freq[cur_code]
        cur_code = ""
print(dec_str)
----------- alternate
k, l = input().split()
dic = {}
for _ in range(int(k)):
    l, code = input().split()
    dic[code] = l[:-1]
    
curr = ""
ans = ""
for i in input():
    curr += i
    if curr in dic:
        ans += dic[curr]
        curr = ""
print(ans)
--------------- alternate
k, l = map(int, input().split())
dic = {}
for _ in range(k):
    key, code = input().split(':')
    dic[key] = code.strip()
s = input()
while len(s) > 0:
    for key in dic:
        if s.startswith(dic[key]):
            print(key, end='', sep='')
            s = s[len(dic[key]):]
--------------- alternate
import sys
lines = [line.rstrip() for line in sys.stdin.readlines()]
table = {}
for line in lines[1:-1]:
    letter, code = line.split(': ')
    table[code] = letter

word = ''
buff = ''
for ch in lines[-1]:
    buff += ch
    letter = table.get(buff, None)
    if letter:
        buff = ''
        word += letter

print(word)
----------- alternate

"""
"""
Теоретическая задача для самостоятельной проверки: свойство кода Хаффмана
Докажите, что если частоты всех символов меньше 1/31/3 (другими словами, каждый символ в исходную строку ss входит строго меньше |s|/3|s|/3 раз), 
то коды всех символов в коде Хаффмана будут длиннее одного бита.
"""

"""
Задача на программирование: очередь с приоритетами



Первая строка входа содержит число операций 1≤n≤1051≤n≤105. Каждая из последующих nn строк задают операцию одного из следующих двух типов:

InsertInsert xx, где 0≤x≤1090≤x≤109 — целое число;
ExtractMaxExtractMax.
Первая операция добавляет число xx в очередь с приоритетами, вторая — извлекает максимальное число и выводит его.
Sample Input:
6
Insert 200
Insert 10
ExtractMax
Insert 5
Insert 500
ExtractMax
Sample Output:
200
500


k=[]

def SwimUP():
	i = len(k)
	if i > 1:
		i -= 1
		while ((i > 0) and (k[i//2] < k[i])):
			t = k[i//2]
			k[i//2] = k[i]
			k[i] = t
			i = i//2

def getmax(i,x):
	if 2*i+1>x and 2*i>x:
		return 0
	else:
		if 2*i+1>x:
			return 2*i
		else:
			return 2*i if k[2*i]>k[2*i+1] else 2*i+1

def SwimDown(x):
	i = 0
	#print('shiftdown',i,k,x)
	#y = getmax(i,x-1)
	#print(k,i,y,x-1)
	#while y>0 and k[i]<k[y]:
	#	#print('ki<kj',k,i,y,k[i],k[y])
	#	t = k[i]
	#	k[i] = k[y]
	#	k[y] = t
	#	i+=1
	#	y = getmax(i,x-1)
	#print('end shiftdown')
	while 2*i<=x-1:
		j=i
		if k[2*i]>k[j]:
			j = 2*i
		if 2*i+1<=x-1 and k[2*i+1]>k[j]:
			j=2*i+1
		if j==i:
			break
		t=k[i]
		k[i]=k[j]
		k[j]=t
		i=j
def Insert(a):
	k.append(a)
	SwimUP()
	#print(k)

def ExtractMax(x):
	t = k[0]
	k[0] = k[x-1]
	#print(k)
	k.pop(x-1)
	#print(k)
	SwimDown(x-1)
	return t

n = int(input())
for i in range(n):
	op = input()
	#print(op)
	if op[0] == 'I':
		Insert(int(op.split()[1]))
	else:
		print(ExtractMax(len(k)))
	#print(k)
	
------------- alternate
class MakeHeap():
    
    def __init__(self):
        self.heap = []

    def shift_up(self):
        i = len(self.heap) - 1
        while self.heap[i] > self.heap[(i-1)//2] and i > 0:
            self.heap[i], self.heap[(i-1)//2] = self.heap[(i-1)//2], self.heap[i]
            i = (i-1)//2

    def shift_down(self):
        i = 1
        try:
            while len(self.heap) >= 2 * i:
                if self.heap[2*i-1] >= self.heap[2*i] and self.heap[2*i-1] > self.heap[i-1]:
                    self.heap[2*i-1], self.heap[i-1] = self.heap[i-1], self.heap[2*i-1]
                    i = 2 * i
                elif self.heap[2*i] > self.heap[2*i-1] and self.heap[2*i] > self.heap[i-1]:
                    self.heap[2*i], self.heap[i-1] = self.heap[i-1], self.heap[2*i]
                    i = 2 * i + 1
                else:
                    break
        except IndexError:
            if len(self.heap) == 2 * i:
                if self.heap[2*i-1] > self.heap[i-1]:
                    self.heap[2*i-1], self.heap[i-1] = self.heap[i-1], self.heap[2*i-1]
        
    def insert_node(self, node):
        if len(self.heap) == 0:
            self.heap.append(node)
        else:
            self.heap.append(node)
            self.shift_up()

    def extract_max(self):
        if len(self.heap) == 1:
            return self.heap.pop()
        else:
            max_node = self.heap[0]
            self.heap[0] = self.heap.pop()
            self.shift_down()
            return max_node


H = MakeHeap()

for _ in range(int(input())):
    command = input().split()
    if command[0] == 'Insert':
        H.insert_node(int(command[1]))
    else:
        print(H.extract_max())
		
------------ alternate
# put your python code here
import heapq
ml = []
heapq.heapify(ml)
n = int(input())
for _ in range(n):
    a = input()
    try:
        c, d = a.split()
        heapq.heappush(ml, -int(d))
    except ValueError:
        c = a
        print(-heapq.heappop(ml))
--------- alternate
import heapq
import sys

input()
h = []
for line in sys.stdin.readlines():
    if 'Insert' in line:
        val = -int(line.split()[1])
        heapq.heappush(h, val)
    elif 'ExtractMax' in line:
        print(-heapq.heappop(h))
"""

"""
Ввод и вывод массива целых чисел разделённых пробелами:

a = list(int(i) for i in input().split())
print(" ".join(map(str, a)))
Если нужно читать много пар чисел, как в этой задаче, то просто:

A = []
for i in range(n):
  a, b = (int(i) for i in input().split())
  A.append((a, b))﻿
  
--------------
Чтобы понять, как работает трюк с reader, нужно узнать про выражения-генераторы (generator expressions) и синтаксис распаковки. 

В вашем примере можно поступить так:

reader = (tuple(map(int, line.split())) for line in input)
n, capacity = next(reader)
[vals_n_weights] = reader  # распакуем единственное значение
или так:

reader = (tuple(map(int, line.split())) for line in input)
(n, capacity), vals_n_weights = reader
-----------------
-- непрерывный рюкзак
import sys

def fractional_knapsack(capacity, values_and_weights):
	order = [(v / w, w) for v,w in values_and_weights]
	order.sort(reverse=True)
	
	acc = 0
	for v_per_w, w in order:
		if w<capacity:
			acc += v_per_w*w
			capacity -= w
		else:
			acc += v_per_w * capacity
			break
	
	return 0

def main():
	reader = (tuple(map(int, line.split())) for line in sys.stdin)
	n,capacity = next(reader)
	values_and_weights = list(reader)
	assert len(values_and_weights) == n
	opt_value = fractional_knapsack(capacity, values_and_weights)
	print("{:.3f}".format(opt_value))
	
if __name__ = "__main__":
	main()
	
------ через двоичную кучу
import heapq
import sys

def fractional_knapsack(capacity, values_and_weights):
	order = [(-v / w, w) for v,w in values_and_weights]
	heapq.heapify(order)
	
	acc = 0
	while order and capacity:
		v_per_w, w = heap.heappop(order)
		can_take = min(w,capacity)
		acc -= v_per_w*can_take
		capacity -= can_take
	
	return 0

def main():
	reader = (tuple(map(int, line.split())) for line in sys.stdin)
	n,capacity = next(reader)
	values_and_weights = list(reader)
	assert len(values_and_weights) == n
	opt_value = fractional_knapsack(capacity, values_and_weights)
	print("{:.3f}".format(opt_value))
	

def test():
	assert fractional_knapsack(0, [(60,20)]) == 0.0
	assert fractional_knapsack(25,[(60,20)]) == 60.0
	assert fractional_knapsack(25, [(60,20), (0,100)]) == 60.0
	assert fractional_knapsack(25, [(60,20),(50,50)] == 60.0 + 5.0
	
	assert fractional_knapsack(50, [(60,20),(100,50),(120,30)] == 180.0

	from random import randint
	from timing import timed
	for attempt in range(100):
		n = randint(1,1000)
		capacity = randint(0,2*10**6)
		values_and_weights = []
		for i in range(n):
			values_and_weights.append((randint(0,2*10**6), randint(1,2*10**6)))
		
		t.timed(fractional_knapsack, capacity, values_and_weights)
		assert t < 5
	
if __name__ = "__main__":
	#main()
	test()
"""

"""
Коды Хаффмана
----------

from collections import Counter, namedtuple
import heapq

class Node(namedtuple("Node", ["left", "right"])):
	def walk(self, code, acc):
		self.left.walk(code, acc + "0")
		self.right.walk(code, acc + "1")
		
class Leaf(namedtuple("Leaf", ["char"])):
	def walk(self, code, acc):
		code[self.char] = acc or "0"

def huffman_encode(s):
	h = []
	for ch, freq in Counter(s).items():
		h.append((freq, len(h), Leaf(ch)))

	heapq.heapify(h)
	
	count = len(h)
	while len(h) > 1:
		freq1, _count1, left = heapq.heappop(h)
		freq2, _count2, right = heapq.heappop(h)
		heapq.heappush(h, (freq1 + freq2, count, Node(left,right)))
		count += 1
	
	code = {}
	if h:
		[(_freq, _count, root)] = h
		root.walk(code, "")
	return code

def main():
	s = input()
	code = huffman_encode(s)
	encoded = "".join(code[ch] for ch in s)
	print(len(code), len(encoded))
	for ch in sorted(code):
		print("{}: {}".format(ch, code[ch]))
	print(encoded)
	
def test(n_iter=100):
	import random
	import string
	
		for i in range(n_iter):
			length= random.randint(0,32)
			s = "".join(random.choice(string.ascii_letters) for _ range(length))
			code = huffman_encode(s)
			encoded = "".join(code[ch[ for ch in s)
			assert hiffman_decode(encoded, code) == s
	
if __name__=="__main__":
	#main()
	test()
	
"""

"""
---------------------------------------------------- экзамен -------------------------------------------------------------------
По данным двум числам 1≤a,b≤2*10^9 найдите наименьшее натуральное число mm, которое делится и на aa, и на bb.

Sample Input:
18 35
Sample Output:
630

#x, y = sorted(map(int, input().split()))
#if y%x==0:
#	print(y)
#else:
#	xx, yy = x, y
#	while x:    
#	    y,  x = x, y % x
#	print(int(xx*yy/y))
a,b = map(int,input().split())
m = a*b
while a != 0 and b != 0:
	if a > b:
		a %= b
	else:
		b %= a
print(m // (a+b))
"""

"""
По данным числам 1≤n≤301≤n≤30 и 1≤w≤1091≤w≤109 и набору чисел 1≤v1,…,vn≤1091≤v1,…,vn≤109 найдите минимальное число kk, 
для которого число ww можно представить как сумму kk чисел из набора {v1,…,vn}{v1,…,vn}. 
Каждое число из набора можно использовать сколько угодно раз. 
Известно, что в наборе есть единица и что для любой пары чисел из набора одно из них делится на другое. 
Гарантируется, что в оптимальном ответе число слагаемых не превосходит 104104.

Выведите число kk и сами слагаемые.

Sample Input:
4 90 1 2 10 50
Sample Output:
5 50 10 10 10 10

x = [int(x) for x in input().split()]
n,w = x[0],x[1]
x.pop(0)
x.pop(0)
#print(n,w,x)
x.sort(reverse=True)
#print(x)
s=[]
while w and x:
	if x[0]<=w:
		s.append(x[0])
		w -= x[0]
	else:
		x.pop(0)
print(len(s)," ".join(map(str,s)))

"""

"""
В первой строке входа дано целое число 2≤n≤2⋅1052≤n≤2⋅105, во второй — последовательность целых чисел0≤a1,a2,…,an≤1050≤a1,a2,…,an≤105. 
Выведите максимальное попарное произведение двух элементов последовательности, то есть max1≤i≠j≤naiajmax1≤i≠j≤naiaj.

Sample Input:
3
1 2 3
Sample Output:
6

n = int(input())
x = [int(x) for x in input().split()]
x.sort(reverse=True)
print(x[0]*x[1])
"""

"""
В первой строке входа дано целое число 3≤n≤2⋅1053≤n≤2⋅105, во второй — последовательность целых чисел 0≤a1,a2,…,an≤1050≤a1,a2,…,an≤105. 
Выведите максимальное произведение трех элементов последовательности, то есть max1≤i<j<k≤naiajakmax1≤i<j<k≤naiajak.

Sample Input:
3
1 2 3
Sample Output:
6

n = int(input())
x = [int(x) for x in input().split()]
x.sort(reverse=True)
print(x[0]*x[1]*x[2])
"""

"""
Дано две строки TT (длиной до 103103) и PP (длиной до 105105). 

Подсчитайте количество точных вхождений второй строки в первую.

Sample Input:
GCGCG
GCG
Sample Output:
2

import re
t = input()
p = input()
patern_p = '(?=(%s))' %  re.escape(p)
print(len(re.findall(patern_p, t)))
------------ alternate
import re
t = input()
p = input()
patern_p = '(?=('+ p +'))'
print(len(re.findall(patern_p, t)))
"""

"""
В первой строке входа дано целое число 3≤n≤2⋅1053≤n≤2⋅105, во второй — последовательность целых чисел 0≤a1,a2,…,an≤1050≤a1,a2,…,an≤105.

Выведите числа массива, соответствующего мин-куче по входным данным.

Sample Input:
7
3 8 4 9 7 5 6
Sample Output:
3 7 4 9 8 5 6


import heapq
k = []
heapq.heapify(k)
n = int(input())
x = [int(x) for x in input().split()]
#print(x)
for i in range(n):
	heapq.heappush(k, x[i])
print(" ".join(map(str, k)))
"""

"""
Удалить максимум из H 	O(log(n))
Увеличить значение в L 	O(n)
Вставить значение в H 	O(log(n))
Удаление из L 			O(1)
Вставить значение в L 	O(n)
Увеличить значение в H 	O(log(n))
"""

http: // younglinux.info / book / export / html / 60

###########################
###########################
# correcting forecast csv
forecasts = open("E:/Docs/GDrive/work/python/forecasts.csv", "r").readlines()

old_value = 1
new_list = []
for f in forecasts[1:]:
    strpf = f.replace('"', '').strip()
    new_str = "%s,%s\n" % (strpf, old_value)
    newspl = new_str.strip().split(",")
    final_str = "%s,%s\n" % (newspl[0], newspl[2])
    final_str = final_str.replace('"', '')
    old_value = f.strip().split(',')[1]
    new_list.append(final_str)

out = open("E:/Docs/GDrive/work/python/forecasts_new.csv", "w")
for n in new_list:
    out.write(n)
###########################
###########################
"""
Нейронный сети
"""
"""
https://www.codecogs.com/latex/eqneditor.php
"""
"""
Источники

При создании курса я, конечно, использовал множество различных источников и опирался на опыт некоторых других онлайн-курсов. 
Ниже приведены те из них, которые просто нельзя не упомянуть перед началом нашего курса.  Данный список будет постепенно пополняться.

https://www.coursera.org/learn/machine-learning/ — живая легенда, курс Andrew Ng по машинному обучению. С него началось когда-то моё увлечение этой темой. 
Крайне рекомендую к просмотру и, пользуясь случаем, хочу выразить публично глубокую благодарность его автору.

http://neuralnetworksanddeeplearning.com/ — замечательная онлайн-книга по нейросетям. Я, кстати, потихоньку её перевожу 
(первые главы должны появиться в открытом доступе в начале лета).

https://www.coursera.org/learn/neural-networks﻿ — я уже использовал фразу «живая легенда»  и теперь испытываю сложности, 
поскольку как-то иначе охарактеризовать Джеффри Хинтона (человека, стоящего у истоков современных подходов к обучению нейросетей 
с помощью алгоритма обратного распространения ошибки) сложно. Курс у него получился отличный.

https://ulearn.azurewebsites.net/Course/AIML/ — недавно обнаружил этот курс, приятные и качественные лекции по широкому набору тем. 
Единственный из источников на русском языке.

http://cs231n.github.io/﻿ — прекрасный курс от Стэнфордского университета, чуть более сложный, пожалуй, чем наш.﻿

"""

"""
import numpy as np
print(2*np.eye(3,4) + np.eye(3,4,1))
"""
"""
Основные методы ndarray

Для работы с многомерными массивами в NumPy реализованы самые часто требующиеся операции. 
Некоторые из них (которые особенно часто будут нужны в нашем курсе) мы сейчас покажем.

Форма массива
a.flatten() — превращает массив в одномерный.
a.T или a.transpose(*axes) — транспонирование (или смена порядка осей в случае, 
когда размерность массива больше двух).
a.reshape(shape) — смена формы массива. Массив "распрямляется" и построчно 
заполняется в новую форму.

>>> import random
>>> w = np.array(random.sample(range(1000), 12)) # одномерный массив из 
12 случайных чисел от 1 до 1000
>>> w = w.reshape((2,2,3)) # превратим w в трёхмерную матрицу
>>> print(w)
[[[536 986 744]
  [543 248 544]]

 [[837 235 415]
  [377 141 751]]]
>>> print(w.transpose(0,2,1))
[[[536 543]
  [986 248]
  [744 544]]

 [[837 377]
  [235 141]
  [415 751]]]
 """
"""
Массив, который нужно было создать в предыдущей задаче, хранится в переменной mat. 
Превратите его в вертикальный вектор и напечатайте.
import numpy as np
mat = mat.reshape((12,1))
print(mat)
"""

"""
Основные методы ndarray

Базовые статистики

a.min(axis=None), a.max(axis=None), a.mean(axis=None), a.std(axis=None) —
 минимум, максимум, среднее арифметическое и стандартное отклонение вдоль указанной оси. 
 По умолчанию ось не указана и статистика считается по всему массиву. a.argmin(axis=None),
 a.argmax(axis=None) — индексы минимального и максимального элемента. 
 Пример:
>>> print(v)
[[1 2 3 4]
 [1 2 3 4]
 [1 2 3 4]]
>>> print(v.mean(axis=0))  # вдоль столбцов
[ 1.  2.  3.  4.]
>>> print(v.mean(axis=1))  # вдоль строк
[ 2.5  2.5  2.5]
>>> print(v.mean(axis=None))  # вдоль всего массива
2.5
Чтобы лучше понять, почему говорят «усреднение вдоль оси» — можно нарисовать 
эту матрицу на бумажке и прямыми линиями соединить те элементы, которые сливаются 
в один при усреднении. Чтобы было совсем понятно — можно ещё добавить координатные оси, 
чтобы каждый элемент wijwij оказался над точкой (i,j)(i,j).
a.sum(axis=None), a.prod(axis=None) — сумма и произведение всех элементов вдоль 
указанной оси. a.cumsum(axis=None), a.cumprod(axis=None) — частичные суммы и произведения
 (для (a1,⋯,an)(a1,⋯,an) вектор частичных сумм — это (a1,a1+a2,⋯,a1+⋯+an)(a1,a1+a2,⋯,a1+⋯+an)).
Линейная алгебра
Пакет numpy.linalg содержит большую часть стандартных операций и разложений матриц. 
Некоторые самые популярные функции вынесены в корень пакета NumPy.
a.dot(b) — матричное произведение двух массивов (размерности должны быть согласованы),
linalg.matrix_power(M, n) — возведение матрицы M в степень n,
a.T — транспонирование
linalg.norm(a, ord=None) — норма матрицы a, по умолчанию норма Фробениуса для матриц 
и L2-норма для векторов; подробное описание возможных норм — в справке,
linalg.inv(a) — матрица, обратная к a (если a необратима, выбрасывается LinAlgError; 
псевдообратная считается через linalg.pinv(a))

>>> a = w.dot([1,2,3])
>>> print(a)
[[4740 2671]
 [2552 2912]]
>>> ainv = np.linalg.inv(a)
>>> print(a.dot(ainv))
[[  1.00000000e+00   0.00000000e+00]
 [ -2.22044605e-16   1.00000000e+00]]

Подробные описания с указанием полного списка аргументов, а также описания всех остальных 
функций находятся на сайте проекта NumPy.
"""
"""
print(linalg.inv((np.eye(3,k=0))))
a.dot(a.T)
print(w.sum(axis=None))
6062

print(w.prod(axis=None))
1571962880

print(w.cumsum(axis=None))
[ 428  505 1002 1190 2065 2947 3277 3389 3606 4144 5098 6062]

a=w.dot([1,2,3])

print(a)
[[2073 4584]
 [1205 5338]]

ainv=np.linalg.inv(a)

print(a.dot(ainv))
[[ 1.  0.]
 [ 0.  1.]]
"""
"""
Задача: перемножьте две матрицы!

На вход программе подаются две матрицы, каждая в следующем формате: 
на первой строке два целых положительных числа nn и mm, разделенных пробелом -
 размерность матрицы. В следующей строке находится n⋅mn⋅m целых чисел, разделенных 
 пробелами - элементы матрицы. Подразумевается, что матрица заполняется построчно, 
 то есть первые mm чисел - первый ряд матрицы, числа от m+1m+1 до 2⋅m2⋅m - второй, и т.д.

Напечатайте произведение матриц XYTXYT, если они имеют подходящую форму, или строку 
"matrix shapes do not match", если формы матриц не совпадают должным образом. 

В этот раз мы проделали за вас подготовительную работу по считыванию матриц (когда 
вы начнёте решать, этот код будет уже написан):

x_shape = tuple(map(int, input().split()))
X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)
 Несколько комментариев:

map(f, iterable, …) — встроенная функция языка Python, возвращает результат поэлементного 
применения функции f к элементам последовательности iterable; если f принимает несколько 
аргументов, то на вход должно быть подано соответствующее число последовательностей: 
результатом map(f, x, y, z) будет итератор, возвращающий поочерёдно 
f(x[0], y[0], z[0]), f(x[1], y[1], z[1]), f(x[2], y[2], z[2]) и так далее; 
результат применения f к очередному набору аргументов вычисляется только тогда, 
когда требуется использовать этот результат, но не ранее, подробнее и короче в справке;
np.fromiter создаёт NumPy-массив из итератора, то есть заставляет итератор вычислить 
все доступные значения и сохраняет их в массив;
input() — встроенная функция языка Python, читает одну строку (последовательность 
символов вплоть до символа переноса строки) из входного потока данных и возвращает 
её как строку;
split() — метод класса string, возвращает список слов в строке (здесь слова — 
последовательности символов, разделённые пробельными символами); принимает дополнительные 
аргументы, подробнее в справке.
Sample Input 1:
2 3
8 7 7 14 4 6
4 3
5 5 1 5 2 6 3 3 9 1 4 6
Sample Output 1:
[[ 82  96 108  78]
 [ 96 114 108  66]]
Sample Input 2:
2 3
5 9 9 10 8 9
3 4
6 11 3 5 4 5 3 2 5 8 2 2
Sample Output 2:
matrix shapes do not match


import numpy as np

x_shape = tuple(map(int, input().split()))
X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)
y_shape = tuple(map(int, input().split()))
Y = np.fromiter(map(int, input().split()), np.int).reshape(y_shape)

# here goes your solution; X and Y are already defined!
if x_shape[1]==y_shape[1]:
	print(X.dot(Y.T))
else:
	print('matrix shapes do not match')
"""

"""
Как считать данные из файла:

>>> sbux = np.loadtxt("sbux.csv", usecols=(0,1,4), skiprows=1, delimiter=",", 
                      dtype={'names': ('date', 'open', 'close'),
                             'formats': ('datetime64[D]', 'f4', 'f4')})
>>> print(sbux[0:4])
[(datetime.date(2015, 9, 1), 53.0, 57.2599983215332)
 (datetime.date(2015, 8, 3), 58.619998931884766, 54.709999084472656)
 (datetime.date(2015, 7, 1), 53.86000061035156, 57.93000030517578)
 (datetime.date(2015, 6, 1), 51.959999084472656, 53.619998931884766)]
 
Здесь использованы не все параметры функции loadtxt (полный их список можно 
посмотреть в справке). Разберём имеющиеся, так как они являются наиболее часто встречающимися.
"sbux.csv" — имя файла (или сюда же можно передать объект файла, такой пример 
вы увидите в следующей задаче урока), из которого считываются данные.
usecols — список колонок, которые нужно использовать. 
Если параметр не указан, считываются все колонки.
skiprows — количество рядов в начале файла, которые нужно пропустить. 
В нашем случае пропущен ряд заголовков. По умолчанию (если значение параметра 
не указано явно) skiprows = 0.
delimiter — разделитель столбцов в одной строке, в csv-файлах это запятая, 
по умолчанию разделителем является любой пробел (в том числе — знак табуляции).
dtype — словарь из названий колонок (переменных) и типов хранящихся в них значений. 
NumPy использует свою собственную систему типов, и названия именно этих типов нужно указать. 
По умолчанию функция попытается самостоятельно угадать, какому типу принадлежат 
подаваемые на вход значения.
"""
"""
Задача: считайте данные из файла и посчитайте их средние значения.

На вход вашему решению будет подан адрес, по которому расположен csv-файл, 
из которого нужно считать данные. Первая строчка в файле — названия столбцов, 
остальные строки — числовые данные (то есть каждая строка, кроме первой, состоит из 
последовательности вещественных чисел, разделённых запятыми).

Посчитайте и напечатайте вектор из средних значений вдоль столбцов входных данных. 
То есть если файл с входными данными выглядит как

a,b,c,d
1.5,3,4,6
2.5,2,7.5,4
3.5,1,3.5,2
то ответом будет

[ 2.5  2.   5.   4. ]
Как упоминалось на предыдущем шаге, в качестве файла для loadtxt можно передать объект файла. 
Это удобно в таких случаях, как сейчас: когда данные лежат не на вашем компьютере, 
а где-то в сети. Как их скачать из сети? С помощью стандартных библиотек:

>>> from urllib.request import urlopen
>>> f = urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')

Теперь в f содержится объект файла, который можно передать в loadtxt.

Sample Input:
https://stepic.org/media/attachments/lesson/16462/boston_houses.csv
Sample Output:
[ 22.53280632   3.61352356  11.36363636   0.06916996   0.55469506
   6.28463439   3.79504269]

   
-------------
from urllib.request import urlopen
import numpy as np
filename = input()
f = urlopen(filename)
sbux = np.loadtxt(f, skiprows=1, delimiter=",")
print(sbux.mean(axis=0))
"""

"""
Теперь давайте попробуем применить наши новые матричные формулы — сначала на 
игрушечном примере, который мы рассматривали пару видео назад.
У нас есть набор данных: знания о длине тормозного пути и скорости для трёх автомобилей.
D	V
10	60
7	50
12	75
Напишите через запятую оценки коэффициентов линейной регрессии D на V, 
т.е. β^0, β^1 для модели D=β0+β1V+ε с точностью до трёх знаков после точки.

import numpy as np
import numpy.linalg as linalg

X=np.array([1,60,1,50,1,75])
print(X.reshape(3,2))
[[ 1 60]
 [ 1 50]
 [ 1 75]]
X=X.reshape(3,2)
Y=np.array([10,7,12])
print(Y.reshape(3,1))
[[10]
 [ 7]
 [12]]
Y=Y.reshape(3,1)

b=((linalg.inv(X.T.dot(X))).dot(X.T)).dot(Y)
print(b)
[[-2.34210526]
 [ 0.19473684]]

"""
"""
Найдите оптимальные коэффициенты для построения линейной регрессии.

На вход вашему решению будет подано название csv-файла, из которого нужно считать данные. 
Пример можно посмотреть здесь. Загрузить их можно следующим образом:

fname = input()  # read file name from stdin
f = urllib.request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with
Ваша задача — подсчитать вектор коэффициентов линейной регрессии для предсказания 
первой переменной (первого столбца данных) по всем остальным. 
Напомним, что модель линейной регрессии — это y=β0+β1x1+⋯+βnxny=β0+β1x1+⋯+βnxn.

Напечатайте коэффициенты линейной регрессии, начиная с β0β0, через пробел. 
Мы будем проверять совпадения с точностью до 4 знаков после запятой.

Методы и приёмы, которые мы ещё не упоминали и которые могут вам помочь в процессе
 решения (могут являться подсказками!):

np.hstack((array1, array2, ...))  # склеивает по строкам массивы, являющиеся компонентами 
кортежа, поданного на вход; массивы должны совпадать по всем измерениям, кроме второго
np.ones_like(array)  # создаёт массив, состоящий из единиц, идентичный по форме массиву array
"delim".join(array)  # возвращает строку, состоящую из элементов array, 
разделённых символами "delim"
map(str, array)  # применяет функцию str к каждому элементу array

Sample Input:
https://stepic.org/media/attachments/lesson/16462/boston_houses.csv
Sample Output:
-3.65580428507 -0.216395502369 0.0737305981755 4.41245057691 -25.4684487841 7.14320155075 -1.30108767765


import urllib
from urllib import request
import numpy as np
import numpy.linalg as linalg

fname = input()  # read file name from stdin
f = urllib.request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with
# here goes your solution
Y=np.array(data[:,0]).reshape(data.shape[0],1)
X=np.array(data)
X[:,0]=1
b = ((linalg.inv(X.T.dot(X))).dot(X.T)).dot(Y)
print(' '.join(map(str, b[:,0])))

--------------- alternate

import numpy as np
import urllib
from urllib import request
f = urllib.request.urlopen(input())
X = np.loadtxt(f, delimiter=',', skiprows=1)
y = np.array([i for i in X[:,0]])
X[:,0] = 1
b = (np.linalg.inv(X.T.dot(X))).dot(X.T.dot(y))
for i in b:
    print(i, end=' ')
------------------ alternate
import urllib
from urllib import request
import numpy as np
from scipy import linalg as la
f = urllib.request.urlopen(input())  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with
y = np.array(data[:,0])
data[:,0] = 1
bet=la.inv(data.T.dot(data)).dot(data.T).dot(y)
print(*bet)
----------------------alternate
import urllib
from urllib import request
import numpy as np
from numpy import linalg as la

fname = input()
if (fname == '') :
	fname = 'https://stepic.org/media/attachments/lesson/16462/boston_houses.csv'
f = urllib.request.urlopen(fname)
data = np.loadtxt(f, delimiter=',', skiprows=1)

y = data[:,0].reshape(data.shape[0],1)
x = np.hstack((np.ones((data.shape[0],1)), data[:,1:]))

b = la.inv(x.T.dot(x)).dot(x.T).dot(y)
print(*b[:,0])

"""

"""
Задание на доказательство (для продвинутых):

Помимо XOR существует множество других задач, с которыми перцептрон не может справиться. К примеру, он не может отличить два паттерна, если мы разрешаем перенос их через край (дальше - подробнее)

Представьте себе две разных бинарных бегущих строки (два разных паттерна): 

Строка типа AA: 

Строка типа BB: 

Мы можем передвигать эти паттерны, перенося через край. При этом тип паттерна считается прежним: 

Можно представить, что мы просто склеиваем края нашей строки, получая непрерывную ленту (кольцо):


Утверждение заключается в том, что перцептрон не может научиться отличать паттерн (кольцо) AA от паттерна BB. Давайте рассмотрим набросок доказательства (тут можно остановить чтение и попробовать доказать самостоятельно, "с нуля"):

Мы бы хотели, чтобы перцептрон всегда выдавал 11 при предъявлении примера, подчиняющегося паттерну AA и 00 при предъявлении примера паттерна BB 

1. Давайте просуммируем значения сумматорной функции на входе перцептрона при предъявлении всевозможных сдвигов паттерна AA. Каждый вес будет учтён столько раз, сколько у нас активных элементов в паттерне AA. В нашем случае, 4 раза. Кроме того, каждый раз при предъявлении примера будет добавляться смещение (bias). Итого,

∑i=1n4⋅wi+n⋅b
∑i=1n4⋅wi+n⋅b
где wiwi - вес для ii-той позиции.
2. Аналогично для паттерна BB.

Придумайте ключевую идею (3 пункт), которая показывает наличие противоречия в первых двух пунктах, если мы предполагаем, что нам удалось правильно классифицировать все примеры.

Пример правильного решения:
Мы хотим получить такие веса, чтобы для паттерна A в любом его предъявлении сумматорная функция выдавала положительное значение (тогда перцептрон для паттерну A будет всегда выдавать 1). Следовательно, если мы сложим значения сумматорной функции на всех вариантах паттерна A, то получим ∑ni=1awi+nb∑i=1nawi+nb, где aa — количество активных клеток в паттерне (у нас 4). И так как сумматорная функция должна быть всегда положительна, ∑ni=1awi+nb>0∑i=1nawi+nb>0.
Для паттерна B в любом его предъявлении сумматорная функция должна выдавать неположительные значения. Следовательно, сумма значений сумматорной функции на всех вариантах паттерна B тоже должна быть неположительна. В паттерне B тоже aa активных клеток, поэтому сумма значений сумматорной функции равна ∑ni=1awi+nb<0∑i=1nawi+nb<0.
Итак,
{∑ni=1awi+nb≤0.
{∑i=1nawi+nb>0.
Это невозможно ни при каких ww и bb, следовательно, набора весов, позволяющего перцептрону разделить два паттерна с одинаковым количеством активных клеток, не существует.
"""

"""
В этом уроке собраны упражнения по программированию для второй недели.

Сперва вам необходимо скачать задания. У вас есть два варианта:

Скачать удобный Jupyter notebook (далее «ноутбук», кто придумает лучше — пишите в комментарии) с картинками, легко читаемыми пояснениями и дополнительными интерактивными визуализациями, которые будут начинать работать по мере выполнения заданий.
Посмотреть на ноутбук из первого пункта в браузере: картинки, пояснения, одна визуализация, но никакой интерактивности и никаких анимаций.
Проблема с ноутбуком только одна — вы можете не знать, что это и как этим пользоваться. Хорошая новость заключается в том, что «учиться» пользоваться jupyter notebook вам сейчас не нужно. Им нужно просто пользоваться =) 
Ладно, две проблемы: ещё у вас может не быть установлен пакет, которым мы пользовались для построения графиков, matplotlib. Это решается легко: conda install matplotlib, pip3 install matplotlib или что-то аналогичное в зависимости от конфигурации вашей системы.

Jupyter notebook — это браузерное приложение, которое вы запускаете на своём компьютере. Вы получаете возможность у себя в браузере читать/писать текст с разметкой (Markdown) и формулами (MathJax, примерно такие же, как на Степике, только не настолько глючные =) ). Но главное — вы можете писать и выполнять код (с подсветкой синтаксиса и подсказками) на Python (и ещё на более чем 40 языках, по словам сайта проекта). Это действительно удобно и много где используется, очень рекомендуем попробовать. 

Весь наш ноутбук состоит из ячеек двух типов: ячейки с кодом и ячейки с текстом. Выполнить ячейку с кодом можно, выделив её и нажав на кнопку play (или Shift+Enter). Чтобы добавить ячейку, выделите какую-нибудь из имеющихся и нажмите B (добавить снизу) или A (сверху). Чтобы удалить — дважды нажмите D. Сменить тип ячейки: Y — на ячейку с кодом,  M — на ячейку с текстом. Чтобы поменять что-то внутри ячейки — нажмите на неё дважды, чтобы войти в режим редактирования. Чтобы выйти из режима редактирования содержания ячейки — нажмите Esc. Интерфейс очень интуитивный и все то же самое можно делать, кликая мышкой по иконкам.

Как установить: conda install jupyter, pip3 install jupyter или что-то аналогичное в зависимости от конфигурации вашей системы. Как запустить: открываете терминал там, где лежит скачанным наш ноутбук, пишете jupyter notebook, жмёте Enter, в вашем браузере открывается список файлов, нажимаете на наш ноутбук и погружаетесь (если слово «терминал» вас пугает - может сработать двойной клик =) ).

Чтобы получить все интерактивные визуализации, вам надо не забывать запускать все клетки с кодом, в которых содержится их описание. Для тех, кто с этим справится, есть отдельное задание на 8 баллов.

Альтернатива - можно весь код перенести в обычный Python скрипт, с минимальными исправлениями всё должно работать. Бонусы: можно крутить графики, которые в ноутбуке статичны. Минусы: что-то может сломаться.

Итак, ссылки:

статически отрисованная версия ноутбука,
репозиторий на GitHub с ноутбуком, данными и картинками.
Задания сформулированы в ноутбуке. Сдавать ваши функции необходимо в соответствующих степах этого урока.

Как и во всём курсе, в ноутбуке код написан на Python3. Небольшие изменения для Python2 предложил @Akarazeev: описание, diff, patch.

Удачи ;)

https://nbviewer.jupyter.org/github/stacymiller/stepic_neural_networks_public/blob/master/HW_1/Hw_1_student_version.ipynb
https://github.com/stacymiller/stepic_neural_networks_public


Реализуйте метод vectorized_forward_pass класса Perceptron. Когда вы начнёте решать задачу, вам нужно будет просто скопировать соответствующую функцию, которую вы написали в ноутбуке (без учёта отступов; шаблон в поле ввода ответа уже будет, ориентируйтесь по нему). Сигнатура функции указана в ноутбуке, она остаётся неизменной.

n — количество примеров, m — количество входов. Размерность входных данных input_matrix — (n, m), размерность вектора весов — (m, 1), смещение (bias) — отдельно. vectorized_forward_pass должен возвращать массив формы (n, 1), состоящий либо из 0 и 1, либо из True и False.

Обратите внимание, необходимо векторизованное решение, то есть без циклов и операторов ветвления. Используйте свойства умножения матриц и возможность эффективно применять какую-нибудь операцию к каждому элементу np.array, чтобы с минимумом кода получить желаемый результат. Например, 

>>> my_vec = np.array([-1, 2, 3]) 
>>> is_positive = my_vec > 0
>>> is_positive
array([False,  True,  True], dtype=bool)


import numpy as np

def vectorized_forward_pass(self, input_matrix):        
    return input_matrix.dot(self.w)+self.b>0
"""

"""
В данном степе вам нужно реализовать метод train_on_single_example класса Perceptron, который получает на вход один набор входных активаций размерности (m,1) и правильный ответ (число 0 или 1), после чего обновляет веса в соответствии с правилом обучения перцептрона. Когда вы начнёте решать задачу, вам нужно будет просто скопировать соответствующую функцию, которую вы написали в ноутбуке (без учёта отступов; шаблон в поле ввода ответа уже будет, ориентируйтесь по нему). Сигнатура функции указана в ноутбуке, она остаётся неизменной.

Обязательно проверяйте размерности на соответствие указанным в задании и в сигнатуре функции!

Дополнительное ограничение: в данной функции нельзя использовать операторы ветвления и циклы. Мы не сможем это проверить во всех случаях (но, возможно, ваше решение с циклом не сможет уложиться в отведённый решению период работы), так что ответственность за выполнение этого ограничения ложится на вашу совесть.

import numpy as np

def train_on_single_example(self, example, y):
    y_ = (self.w.T.dot(example) + self.b)>0
    self.w += (y - y_)*example
    self.b += (y - y_)
    return (y-y_)

--------- alternate
def train_on_single_example(self, example, y):
    answer = self.forward_pass(example)
    diff = y - answer
    self.b += diff
    self.w += example * diff
    return abs(diff)
"""

"""
Реализуйте методы activation и summatory класса Neuron. Когда вы начнёте решать задачу, вам нужно будет просто скопировать соответствующую функцию, которую вы написали в ноутбуке (без учёта отступов; шаблон в поле ввода ответа уже будет, ориентируйтесь по нему). Сигнатура функции указана в ноутбуке, она остаётся неизменной.

n — количество примеров, m — количество входов. Размерность входных данных input_matrix — (n, m), размерность вектора весов — (m, 1). vectorized_forward_pass должен возвращать массив формы (n, 1), состоящий из чисел (float). Мы будем проверять именно правильность ответа, который возвращает vectorized_forward_pass.
"""

Jupyter
NotebookLogout
neural_networks
Last
Checkpoint: 18
hours
ago(autosaved)
Python
3
File
Edit
View
Insert
Cell
Kernel
Widgets
Help
CellToolbar
In[1]:

import matplotlib.pyplot as plt
import numpy as np
import random
​
​
random.seed(42)
In[2]:

% matplotlib
inline
data = np.loadtxt("data.csv", delimiter=",")
pears = data[:, 2] == 1
apples = np.logical_not(pears)
plt.scatter(data[apples][:, 0], data[apples][:, 1], color="red")
plt.scatter(data[pears][:, 0], data[pears][:, 1], color="green")
plt.xlabel("yellowness")
plt.ylabel("symmetry")
plt.show()
% matplotlib
inline
data = np.loadtxt("data.csv", delimiter=",")
pears = data[:, 2] == 1
apples = np.logical_not(pears)
plt.scatter(data[apples][:, 0], data[apples][:, 1], color="red")
plt.scatter(data[pears][:, 0], data[pears][:, 1], color="green")
plt.xlabel("yellowness")
plt.ylabel("symmetry")
plt.show()

In[3]:

my_vec = np.array([-1, 2, 3])
is_positive = my_vec > 0
is_positive
Out[3]:
array([False, True, True], dtype=bool)
In[4]:


class Perceptron:

    ​

def __init__(self, w, b):
    """
    Инициализируем наш объект - перцептрон.
    w - вектор весов размера (m, 1), где m - количество переменных
    b - число
    """

    self.w = w
    self.b = b

​

def forward_pass(self, single_input):
    """
    Метод рассчитывает ответ перцептрона при предъявлении одного примера
    single_input - вектор примера размера (m, 1).
    Метод возвращает число (0 или 1) или boolean (True/False)
    """

    result = 0
    for i in range(0, len(self.w)):
        result += self.w[i] * single_input[i]
    result += self.b

    if result > 0:
        return 1
    else:
        return 0


def vectorized_forward_pass(self, input_matrix):
    return input_matrix.dot(self.w) + self.b > 0


def train_on_single_example(self, example, y):
    y_ = (self.w.T.dot(example) + self.b) > 0
    self.w += (y - y_) * example
    self.b += (y - y_)
    return (y - y_)


def train_until_convergence(self, input_matrix, y, max_steps=1e8):
    """
    input_matrix - матрица входов размера (n, m),
    y - вектор правильных ответов размера (n, 1) (y[i] - правильный ответ на пример input_matrix[i]),
    max_steps - максимальное количество шагов.
    Применяем train_on_single_example, пока не перестанем ошибаться или до умопомрачения.
    Константа max_steps - наше понимание того, что считать умопомрачением.
    """
    i = 0
    errors = 1
    while errors and i < max_steps:
        i += 1
        errors = 0
        for example, answer in zip(input_matrix, y):
            example = example.reshape((example.size, 1))
            error = self.train_on_single_example(example, answer)
            errors += int(error)  # int(True) = 1, int(False) = 0, так что можно не делать if

​
In[5]:


def plot_line(coefs):
    """
    рисует разделяющую прямую, соответствующую весам, переданным в coefs = (weights, bias), 
    где weights - ndarray формы (2, 1), bias - число
    """
    w, bias = coefs
    a, b = - w[0][0] / w[1][0], - bias / w[1][0]
    xx = np.linspace(*plt.xlim())
    line.set_data(xx, a * xx + b)


In[6]:


def step_by_step_weights(p, input_matrix, y, max_steps=1e6):
    """
    обучает перцептрон последовательно на каждой строчке входных данных, 
    возвращает обновлённые веса при каждом их изменении
    p - объект класса Perceptron
    """
    i = 0
    errors = 1
    while errors and i < max_steps:
        i += 1
        errors = 0
        for example, answer in zip(input_matrix, y):
            example = example.reshape((example.size, 1))

            error = p.train_on_single_example(example, answer)
            errors += error  # здесь мы упадём, если вы забыли вернуть размер ошибки из train_on_single_example
            if error:  # будем обновлять положение линии только тогда, когда она изменила своё положение
                yield p.w, p.b

    for _ in range(20): yield p.w, p.b


In[7]:

import matplotlib.pyplot as plt
import numpy as np
import random
    ​
    ​
random.seed(42)  # начальное состояние генератора случайных чисел, чтобы можно было воспроизводить результаты.
​
% matplotlib
inline
data = np.loadtxt("data.csv", delimiter=",")
pears = data[:, 2] == 1
apples = np.logical_not(pears)
plt.scatter(data[apples][:, 0], data[apples][:, 1], color="red")
plt.scatter(data[pears][:, 0], data[pears][:, 1], color="green")
plt.xlabel("yellowness")
plt.ylabel("symmetry")
plt.show()

In[8]:


def create_perceptron(m):
    """Создаём перцептрон со случайными весами и m входами"""
    w = np.random.random((m, 1))
    return Perceptron(w, 1)


In[9]:


def test_v_f_p(n, m):
    """
    Расчитывает для перцептрона с m входами
    с помощью методов forward_pass и vectorized_forward_pass
    n ответов перцептрона на случайных данных.
    Возвращает время, затраченное vectorized_forward_pass и forward_pass
    на эти расчёты.
    """

    p = create_perceptron(m)
    input_m = np.random.random_sample((n, m))

    start = time.clock()
    vec = p.vectorized_forward_pass(input_m)
    end = time.clock()
    vector_time = end - start

    start = time.clock()
    for i in range(0, n):
        p.forward_pass(input_m[i])
    end = time.clock()
    plain_time = end - start

​
return [vector_time, plain_time]
In[10]:


def mean_execution_time(n, m, trials=100):
    """среднее время выполнения forward_pass и vectorized_forward_pass за trials испытаний"""

    return np.array([test_v_f_p(m, n) for _ in range(trials)]).mean(axis=0)

​

def plot_mean_execution_time(n, m):
    """рисует графики среднего времени выполнения forward_pass и vectorized_forward_pass"""

    mean_vectorized, mean_plain = mean_execution_time(int(n), int(m))
    p1 = plt.bar([0], mean_vectorized, color='g')
    p2 = plt.bar([1], mean_plain, color='r')

​
plt.ylabel("Time spent")
plt.yticks(np.arange(0, mean_plain))
​
plt.xticks(range(0, 1))
plt.legend(("vectorized", "non - vectorized"))
​
plt.show()
​
interact(plot_mean_execution_time,
         n=RadioButtons(options=["1", "10", "100"]),
         m=RadioButtons(options=["1", "10", "100"], separator=" "));

In[11]:

% matplotlib
nbagg
​
np.random.seed(1)
fig = plt.figure()
plt.scatter(data[apples][:, 0], data[apples][:, 1], color="red", marker=".", label="Apples")
plt.scatter(data[pears][:, 0], data[pears][:, 1], color="green", marker=".", label="Pears")
plt.xlabel("yellowness")
plt.ylabel("symmetry")
line, = plt.plot([], [], color="black", linewidth=2)  # создаём линию, которая будет показывать границу разделения
​
from matplotlib.animation import FuncAnimation
​
perceptron_for_weights_line = create_perceptron(2)  # создаём перцептрон нужной размерности со случайными весами
​
from functools import partial

weights_ani = partial(
    step_by_step_weights, p=perceptron_for_weights_line, input_matrix=data[:, :-1], y=data[:, -1][:, np.newaxis]
)  # про partial почитайте на https://docs.python.org/3/library/functools.html#functools.partial
​
ani = FuncAnimation(fig, func=plot_line, frames=weights_ani, blit=False, interval=10, repeat=True)
# если Jupyter не показывает вам анимацию - раскомментируйте строчку ниже и посмотрите видео
# ani.save("perceptron_seeking_for_solution.mp4", fps=15)
plt.show()
​
## Не забудьте остановить генерацию новых картинок, прежде чем идти дальше (кнопка "выключить" в правом верхнем углу графика)


In[12]:


def step_by_step_errors(p, input_matrix, y, max_steps=1e6):
    """
    обучает перцептрон последовательно на каждой строчке входных данных, 
    на каждом шаге обучения запоминает количество неправильно классифицированных примеров
    и возвращает список из этих количеств
    """

    def count_errors():
        return np.abs(p.vectorized_forward_pass(input_matrix).astype(np.int) - y).sum()

    errors_list = [count_errors()]
    i = 0
    errors = 1
    while errors and i < max_steps:
        i += 1
        errors = 0
        for example, answer in zip(input_matrix, y):
            example = example.reshape((example.size, 1))

            error = p.train_on_single_example(example, answer)
            errors += error
            errors_list.append(count_errors())
    return errors_list


In[13]:

% matplotlib
inline
perceptron_for_misclassification = create_perceptron(2)
errors_list = step_by_step_errors(perceptron_for_misclassification, input_matrix=data[:, :-1],
                                  y=data[:, -1][:, np.newaxis])
plt.plot(errors_list);
plt.ylabel("Number of errors")
plt.xlabel("Algorithm step number");

In[14]:


def get_vector(p):
    """возвращает вектор из всех весов перцептрона, включая смещение"""
    v = np.array(list(p.w.ravel()) + [p.b])
    return v


In[15]:


def step_by_step_distances(p, ideal, input_matrix, y, max_steps=1e6):
    """обучает перцептрон p и записывает каждое изменение расстояния от текущих весов до ideal"""
    distances = [norm(get_vector(p) - ideal)]
    i = 0
    errors = 1
    while errors and i < max_steps:
        i += 1
        errors = 0
        for example, answer in zip(input_matrix, y):
            example = example.reshape((example.size, 1))

            error = p.train_on_single_example(example, answer)
            errors += error
            if error:
                distances.append(norm(get_vector(p) - ideal))
    return distances


In[16]:

% matplotlib
inline
​
np.random.seed(42)
init_weights = np.random.random_sample(3)
w, b = init_weights[:-1].reshape((2, 1)), init_weights[-1]
ideal_p = Perceptron(w.copy(), b.copy())
ideal_p.train_until_convergence(data[:, :-1], data[:, -1][:, np.newaxis])
ideal_weights = get_vector(ideal_p)
​
new_p = Perceptron(w.copy(), b.copy())
distances = step_by_step_distances(new_p, ideal_weights, data[:, :-1], data[:, -1][:, np.newaxis])
​
plt.xlabel("Number of weight updates")
plt.ylabel("Distance between good and current weights")
plt.plot(distances);

In[17]:

## Определим разные полезные функции
​

def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))

​

def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))


In[18]:


class Neuron:

    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):
        """
        weights - вертикальный вектор весов нейрона формы (m, 1), weights[0][0] - смещение
        activation_function - активационная функция нейрона, сигмоидальная функция по умолчанию
        activation_function_derivative - производная активационной функции нейрона
        """

        assert weights.shape[1] == 1, "Incorrect weight shape"

        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward_pass(self, single_input):
        """
        активационная функция логистического нейрона
        single_input - вектор входов формы (m, 1), 
        первый элемент вектора single_input - единица (если вы хотите учитывать смещение)
        """

        result = 0
        for i in range(self.w.size):
            result += float(self.w[i] * single_input[i])
        return self.activation_function(result)

    def summatory(self, input_matrix):
        """
        Вычисляет результат сумматорной функции для каждого примера из input_matrix. 
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вектор значений сумматорной функции размера (n, 1).
        """
        # Этот метод необходимо реализовать
        return input_matrix.dot(self.w)

​

def activation(self, summatory_activation):
    """
    Вычисляет для каждого примера результат активационной функции,
    получив на вход вектор значений сумматорной функций
    summatory_activation - вектор размера (n, 1),
    где summatory_activation[i] - значение суммматорной функции для i-го примера.
    Возвращает вектор размера (n, 1), содержащий в i-й строке
    значение активационной функции для i-го примера.
    """
    # Этот метод необходимо реализовать

    return self.activation_function(summatory_activation)


def vectorized_forward_pass(self, input_matrix):
    """
    Векторизованная активационная функция логистического нейрона.
    input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
    n - количество примеров, m - количество переменных.
    Возвращает вертикальный вектор размера (n, 1) с выходными активациями нейрона
    (элементы вектора - float)
    """
    return self.activation(self.summatory(input_matrix))


def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
    """
    Внешний цикл алгоритма градиентного спуска.
    X - матрица входных активаций (n, m)
    y - вектор правильных ответов (n, 1)

    learning_rate - константа скорости обучения
    batch_size - размер батча, на основании которого
    рассчитывается градиент и совершается один шаг алгоритма

    eps - критерий остановки номер один: если разница между значением целевой функции
    до и после обновления весов меньше eps - алгоритм останавливается.
    Вторым вариантом была бы проверка размера градиента, а не изменение функции,
    что будет работать лучше - неочевидно. В заданиях используйте первый подход.

    max_steps - критерий остановки номер два: если количество обновлений весов
    достигло max_steps, то алгоритм останавливается

    Метод возвращает 1, если отработал первый критерий остановки (спуск сошёлся)
    и 0, если второй (спуск не достиг минимума за отведённое время).
    """

    # Этот метод необходимо реализовать
    step = 0
    n = X.shape[0]
    res = 0
    a = np.arange(n)
    while (step < max_steps) & (res == 0):
        np.random.shuffle(a)
        rnd = np.random.choice(a, batch_size, replace=False)
        newX = X[rnd]
        newY = y[rnd]
        res = self.update_mini_batch(newX, newY, learning_rate, eps)
        step += 1
    return res


def update_mini_batch(self, X, y, learning_rate, eps):
    """
    X - матрица размера (batch_size, m)
    y - вектор правильных ответов размера (batch_size, 1)
    learning_rate - константа скорости обучения
    eps - критерий остановки номер один: если разница между значением целевой функции
    до и после обновления весов меньше eps - алгоритм останавливается.

    Рассчитывает градиент (не забывайте использовать подготовленные заранее внешние функции)
    и обновляет веса нейрона. Если ошибка изменилась меньше, чем на eps - возвращаем 1,
    иначе возвращаем 0.
    """
    # Этот метод необходимо реализовать

    # loss before update
    # loss0 = 0.5 * np.mean((self.vectorized_forward_pass(X) - y) ** 2)
    # update weights
    # Xout = self.summatory(X)
    # activ = self.activation_function(Xout)
    # deriv = self.activation_function_derivative(Xout)
    # grad = np.dot(np.transpose((activ - y) * deriv), X) / (X.shape[0])
    # self.w = self.w - learning_rate * grad.T
    # loss after update
    # loss1 = 0.5 * np.mean((self.vectorized_forward_pass(X) - y) ** 2)
    # return 1 if abs(loss1 - loss0) < eps else 0

    ##### alternative my
    grad = compute_grad_analytically(self, X, y)
    y0 = J_quadratic(self, X, y)  # 0.5 * np.mean((self.vectorized_forward_pass(X) - y) ** 2)
    self.w -= learning_rate * grad
    y1 = J_quadratic(self, X, y)  # 0.5 * np.mean((self.vectorized_forward_pass(X) - y) ** 2)
    delta = y1 - y0
    return 1 if abs(delta) < eps else 0

# --------- alternate
# def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
#    for i in range(max_steps):
#        sample_ids = np.random.choice(range(X.shape[0]), batch_size, replace=False)
#        Xsample = X[sample_ids, :]
#        ysample = y[sample_ids]
#        if self.update_mini_batch(Xsample, ysample, learning_rate, eps):
#            return 1
#    return 0
​

# def update_mini_batch(self, X, y, learning_rate, eps):
#    loss0 = J_quadratic(self, X, y)
#    grad = compute_grad_analytically(self, X, y)
#    self.w = self.w - learning_rate * grad
#    loss1 = J_quadratic(self, X, y)
#    return 1 if abs(loss1 - loss0) < eps else 0

def J_quadratic(neuron, X, y):
    """
    Оценивает значение квадратичной целевой функции.
    Всё как в лекции, никаких хитростей.
​
    neuron - нейрон, у которого есть метод vectorized_forward_pass, предсказывающий значения на выборке X
    X - матрица входных активаций (n, m)
    y - вектор правильных ответов (n, 1)
​
    Возвращает значение J (число)
    """

​
assert y.shape[1] == 1, 'Incorrect y shape'
​
return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)
​

def J_quadratic_derivative(y, y_hat):
    """
    Вычисляет вектор частных производных целевой функции по каждому из предсказаний.
    y_hat - вертикальный вектор предсказаний,
    y - вертикальный вектор правильных ответов,
​
    В данном случае функция смехотворно простая, но если мы захотим поэкспериментировать 
    с целевыми функциями - полезно вынести эти вычисления в отдельный этап.
​
    Возвращает вектор значений производной целевой функции для каждого примера отдельно.
    """

​
assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'
​
return (y_hat - y) / len(y)
​

def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
    """
    Аналитическая производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для примеров из матрицы X
    J_prime - функция, считающая производные целевой функции по ответам
​
    Возвращает вектор размера (m, 1)
    """

​
# Вычисляем активации
# z - вектор результатов сумматорной функции нейрона на разных примерах
​
z = neuron.summatory(X)
y_hat = neuron.activation(z)
​
# Вычисляем нужные нам частные производные
dy_dyhat = J_prime(y, y_hat)
dyhat_dz = neuron.activation_function_derivative(z)
​
# осознайте эту строчку:
dz_dw = X
​
# а главное, эту:
grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)
​
# можно было написать в два этапа. Осознайте, почему получается одно и то же
# grad_matrix = dy_dyhat * dyhat_dz * dz_dw
# grad = np.sum(, axis=0)
​
# Сделаем из горизонтального вектора вертикальный
grad = grad.T
​
return grad


def compute_grad_numerically(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для тестовой выборки X
    J - целевая функция, градиент которой мы хотим получить
    eps - размер $\delta w$ (малого изменения весов)
    """

​
initial_cost = J(neuron, X, y)
w_0 = neuron.w
num_grad = np.zeros(w_0.shape)
​
for i in range(len(w_0)):
​
old_wi = neuron.w[i].copy()
# Меняем вес
neuron.w[i] += eps
​
# Считаем новое значение целевой функции и вычисляем приближенное значение градиента
num_grad[i] = (J(neuron, X, y) - initial_cost) / eps
​
# Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
neuron.w[i] = old_wi
​
# проверим, что не испортили нейрону веса своими манипуляциями
assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
return num_grad
In[19]:

np.random.seed(42)
n = 10
m = 5
​
X = 20 * np.random.sample((n, m)) - 10
y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]
w = 2 * np.random.random((m, 1)) - 1
​
neuron = Neuron(w)
neuron.update_mini_batch(X, y, 0.1, 1e-5)
# neuron.SGD(X, y, 2, 0.1, 1e-5)
Out[19]:
0
In[20]:

print(neuron.w)
[[-0.22368982]
 [-0.45599204]
 [0.65727411]
 [-0.28380677]
 [-0.43011026]]
In[35]:

# Подготовим данные
​
X = data[:, :-1]
y = data[:, -1]
​
X = np.hstack((np.ones((len(y), 1)), X))
y = y.reshape((len(y), 1))  # Обратите внимание на эту очень противную и важную строчку
​
​
# Создадим нейрон
​
w = np.random.random((X.shape[1], 1))
neuron = Neuron(w, activation_function=sigmoid, activation_function_derivative=sigmoid_prime)
​
# Посчитаем пример
num_grad = compute_grad_numerically(neuron, X, y, J=J_quadratic)
an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)
​
print("Численный градиент: \n", num_grad)
print("Аналитический градиент: \n", an_grad)
Численный
градиент:
[[0.02517702]
 [0.00028275]
 [0.0283814]]
Аналитический
градиент:
[[0.02361927]
 [-0.00074347]
 [0.02855421]]
In[36]:


def print_grad_diff(eps):
    num_grad = compute_grad_numerically(neuron, X, y, J=J_quadratic, eps=float(eps))
    an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)
    print(np.linalg.norm(num_grad - an_grad))


interact(print_grad_diff,
         eps=RadioButtons(options=["3", "1", "0.1", "0.001", "0.0001"]), separator=" ");
0.0146060061038
In[37]:


def compute_grad_numerically_2(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции.
    neuron - объект класса Neuron с вертикальным вектором весов w,
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений,
    y - правильные ответы для тестовой выборки X,
    J - целевая функция, градиент которой мы хотим получить,
    eps - размер $\delta w$ (малого изменения весов).
    """

    # эту функцию необходимо реализовать
    initial_cost = J(neuron, X, y)
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)

​
for i in range(len(w_0)):
​
old_wi = neuron.w[i].copy()
# Меняем вес на +
neuron.w[i] += eps
J_plus_delta = J(neuron, X, y)
# Меняем вес на -
neuron.w[i] -= eps
neuron.w[i] -= eps
J_minus_delta = J(neuron, X, y)
# Считаем новое значение целевой функции и вычисляем приближенное значение градиента
num_grad[i] = (J_plus_delta - J_minus_delta) / (2 * eps)
​
# Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
neuron.w[i] = old_wi
​
# проверим, что не испортили нейрону веса своими манипуляциями
assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
return num_grad

# ------------ alternate
# w_0 = neuron.w
# num_grad = np.zeros(w_0.shape)
# for i in range(len(w_0)):
#    old_wi = neuron.w[i].copy()
#
#    neuron.w[i] = old_wi + eps
#    plus_cost = J(neuron, X, y)
#
#    neuron.w[i] = old_wi - eps
#    minus_cost = J(neuron, X, y)
#
#    num_grad[i] = (plus_cost - minus_cost)/(2*eps)
#    neuron.w[i] = old_wi
# return num_grad
In[]:

​
In[38]:


def print_grad_diff_2(eps):
    num_grad = compute_grad_numerically_2(neuron, X, y, J=J_quadratic, eps=float(eps))
    an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)
    print(np.linalg.norm(num_grad - an_grad))


interact(print_grad_diff_2,
         eps=RadioButtons(options=["3", "1", "0.1", "0.001", "0.0001"]), separator=" ");
0.0334794558243
In[39]:


def J_by_weights(weights, X, y, bias):
    """
    Посчитать значение целевой функции для нейрона с заданными весами.
    Только для визуализации
    """
    new_w = np.hstack((bias, weights)).reshape((3, 1))
    return J_quadratic(Neuron(new_w), X, y)


In[26]:

# %matplotlib qt4
% matplotlib
inline
​
max_b = 40
min_b = -40
max_w1 = 40
min_w1 = -40
max_w2 = 40
min_w2 = -40
​
g_bias = 0  # график номер 2 будет при первой генерации по умолчанию иметь то значение b, которое выставлено в первом
X_corrupted = X.copy()
y_corrupted = y.copy()
​

@interact(fixed_bias=FloatSlider(min=min_b, max=max_b, continuous_update=False),
          mixing=FloatSlider(min=0, max=1, continuous_update=False, value=0),
          shifting=FloatSlider(min=0, max=1, continuous_update=False, value=0)
          )
def visualize_cost_function(fixed_bias, mixing, shifting):
    """
    Визуализируем поверхность целевой функции на (опционально) подпорченных данных и сами данные.
    Портим данные мы следующим образом: сдвигаем категории навстречу друг другу, на величину, равную shifting 
    Кроме того, меняем классы некоторых случайно выбранных примеров на противоположнее.
    Доля таких примеров задаётся переменной mixing
    
    Нам нужно зафиксировать bias на определённом значении, чтобы мы могли что-нибудь визуализировать.
    Можно посмотреть, как bias влияет на форму целевой функции
    """
    xlim = (min_w1, max_w1)
    ylim = (min_w2, max_w2)
    xx = np.linspace(*xlim, num=101)
    yy = np.linspace(*ylim, num=101)
    xx, yy = np.meshgrid(xx, yy)
    points = np.stack([xx, yy], axis=2)

    # не будем портить исходные данные, будем портить их копию
    corrupted = data.copy()

    # инвертируем ответы для случайно выбранного поднабора данных
    mixed_subset = np.random.choice(range(len(corrupted)), int(mixing * len(corrupted)), replace=False)
    corrupted[mixed_subset, -1] = np.logical_not(corrupted[mixed_subset, -1])

    # сдвинем все груши (внизу справа) на shifting наверх и влево
    pears = corrupted[:, 2] == 1
    apples = np.logical_not(pears)
    corrupted[pears, 0] -= shifting
    corrupted[pears, 1] += shifting

    # вытащим наружу испорченные данные
    global X_corrupted, y_corrupted
    X_corrupted = np.hstack((np.ones((len(corrupted), 1)), corrupted[:, :-1]))
    y_corrupted = corrupted[:, -1].reshape((len(corrupted), 1))

    # посчитаем значения целевой функции на наших новых данных
    calculate_weights = partial(J_by_weights, X=X_corrupted, y=y_corrupted, bias=fixed_bias)
    J_values = np.apply_along_axis(calculate_weights, -1, points)

    fig = plt.figure(figsize=(16, 5))
    # сначала 3D-график целевой функции
    ax_1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax_1.plot_surface(xx, yy, J_values, alpha=0.3)
    ax_1.set_xlabel("$w_1$")
    ax_1.set_ylabel("$w_2$")
    ax_1.set_zlabel("$J(w_1, w_2)$")
    ax_1.set_title("$J(w_1, w_2)$ for fixed bias = ${}$".format(fixed_bias))
    # потом плоский поточечный график повреждённых данных
    ax_2 = fig.add_subplot(1, 2, 2)
    plt.scatter(corrupted[apples][:, 0], corrupted[apples][:, 1], color="red", alpha=0.7)
    plt.scatter(corrupted[pears][:, 0], corrupted[pears][:, 1], color="green", alpha=0.7)
    ax_2.set_xlabel("yellowness")
    ax_2.set_ylabel("symmetry")
    ax_1.scatter(10, -10, 0.1)

​
plt.show()

In[44]:


@interact(b=BoundedFloatText(value=str(g_bias), min=min_b, max=max_b, description="Enter $b$:"),
          w1=BoundedFloatText(value="0", min=min_w1, max=max_w1, description="Enter $w_1$:"),
          w2=BoundedFloatText(value="0", min=min_w2, max=max_w2, description="Enter $w_2$:"),
          learning_rate=Dropdown(options=["0.01", "0.05", "0.1", "0.5", "1", "5", "10"],
                                 value="0.01", description="Learning rate: ")
          )
def learning_curve_for_starting_point(b, w1, w2, learning_rate=0.01):
    w = np.array([b, w1, w2]).reshape(X_corrupted.shape[1], 1)
    # learning_rate=float(learning_rate)
    learning_rate = 30
    neuron = Neuron(w, activation_function=sigmoid, activation_function_derivative=sigmoid_prime)

​
story = [J_quadratic(neuron, X_corrupted, y_corrupted)]
for _ in range(2000):
    neuron.SGD(X_corrupted, y_corrupted, 2, learning_rate=learning_rate, max_steps=2)
    story.append(J_quadratic(neuron, X_corrupted, y_corrupted))
plt.plot(story)

plt.title("Learning curve.\n Final $b={0:.3f}$, $w_1={1:.3f}, w_2={2:.3f}$".format(*neuron.w.ravel()))
plt.ylabel("$J(w_1, w_2)$")
plt.xlabel("Weight and bias update number")
plt.show()

In[]:

"""
Итак, мы знаем, как посчитать «назад» ошибку из l+1l+1 слоя в ll-й. Чтобы это знание не утекло куда подальше, давайте сразу его запрограммируем. Заодно вспомним различия между .dot и *.

Напишите функцию, которая, используя набор ошибок δl+1δl+1 для nn примеров, матрицу весов Wl+1Wl+1 и набор значений сумматорной функции на ll-м шаге для этих примеров, возвращает значение ошибки δlδl на ll-м слое сети.

Сигнатура: get_error(deltas, sums, weights), где deltas — ndarray формы (nn, nl+1nl+1), содержащий в ii-й строке значения ошибок для ii-го примера из входных данных, sums — ndarray формы (nn, nlnl), содержащий в ii-й строке значения сумматорных функций нейронов ll-го слоя для ii-го примера из входных данных, weights — ndarray формы (nl+1nl+1, nlnl), содержащий веса для перехода между ll-м и l+1l+1-м слоем сети. Требуется вернуть вектор δlδl — ndarray формы (nlnl, 1); мы не проверяем размер (форму) ответа, но это может помочь вам сориентироваться. Все нейроны в сети — сигмоидальные. Функции sigmoid и sigmoid_prime уже определены.

Обратите внимание, в предыдущей задаче мы работали только с одним примером, а сейчас вам на вход подаётся несколько. Не забудьте учесть этот факт и просуммировать всё, что нужно. И разделить тоже. Подсказка: J=1n∑ni=112∣∣y^(i)−y(i)∣∣2⟹∂J∂θ=1n∑ni=1∂∂θ(12∣∣y^(i)−y(i)∣∣2)J=1n∑i=1n12|y^(i)−y(i)|2⟹∂J∂θ=1n∑i=1n∂∂θ(12|y^(i)−y(i)|2) для любого параметра θθ, который не число примеров.
"""

    def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    # here goes your code
    n = deltas.shape[0]
    return sum((deltas.dot(weights)) * sigmoid_prime(sums)) / n


"""
seaborn
heatmap

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10,12)
ax = sns.heatmap(uniform_data)


ax = sns.heatmap(uniform_data, vmin=0, vmax=1)


Застревание в локальном минимуме => Попробовать несколько начальных точек (начальных весов) сети

Веса и целевая функция почти/вообще не меняются => Нормировать входы

Переобучение =>Устроить кросс-валидацию

Форма изменения целевой функции хорошая, но обучение протекает долго=>Увеличить learning rate

δL=∇aLJ⊙(aL)′(zL)δL=∇aLJ⊙(aL)′(zL)
δl=((Wl+1)Tδl+1)⊙(al)′(zl)δl=((Wl+1)Tδl+1)⊙(al)′(zl)
∂J∂blj=δlj
"""

"""
http://neuralnetworksanddeeplearning.com/chap2.html
http://statweb.stanford.edu/~tibs/ElemStatLearn/
http://localhost:8888/notebooks/Hw_2_student_version.ipynb
"""

###########################
###########################
"""
Given a square matrix of size , calculate the absolute difference between the sums of its diagonals.
Input Format
The first line contains a single integer, . The next  lines denote the matrix's rows, with each line containing space-separated integers describing the columns.
Print the absolute difference between the two sums of the matrix's diagonals as a single integer.

#!/bin/python3

import sys

def diagonalDifference(a):
    # Complete this function
    p = 0
    s = 0
    for i in range(len(a)):
        p += a[i][i]
        s += a[i][(len(a)-1-i)]
    return(abs(p-s))

if __name__ == "__main__":
    n = int(input().strip())
    a = []
    for a_i in range(n):
       a_t = [int(a_temp) for a_temp in input().strip().split(' ')]
       a.append(a_t)
    result = diagonalDifference(a)
    print(result)
"""
"""
Given an array of integers, calculate which fraction of its elements are positive, which fraction of its elements are negative, and which fraction of its elements are zeroes, respectively. Print the decimal value of each fraction on a new line.

#!/bin/python3

import sys

def plusMinus(arr):
    # Complete this function
    a = [0,0,0]
    if len(arr)>0:
        for i in arr:
            a[0] += 1 if i>0 else 0
            a[1] += 1 if i<0 else 0
            a[2] += 1 if i==0 else 0
        print("{0:.6f}".format(a[0]/len(arr)))
        print("{0:.6f}".format(a[1]/len(arr)))
        print("{0:.6f}".format(a[2]/len(arr)))
    else:
        print(0)
        print(0)
        print(0)

if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    plusMinus(arr)
"""

"""
Consider a staircase of size :

   #
  ##
 ###
####
Observe that its base and height are both equal to , and the image is drawn using # symbols and spaces. The last line is not preceded by any spaces.

Write a program that prints a staircase of size .

#!/bin/python3

import sys

def staircase(n):
    # Complete this function
    for i in range(n):
        print(' '*(n-i-1)+'#'*(i+1))
if __name__ == "__main__":
    n = int(input().strip())
    staircase(n)
"""
"""
Given five positive integers, find the minimum and maximum values that can be calculated by summing exactly four of the five integers. Then print the respective minimum and maximum values as a single line of two space-separated long integers.

#!/bin/python3

import sys

def miniMaxSum(arr):
    # Complete this function
    arrs = sorted(arr)
    print(sum(arrs[0:4]),sum(arrs[1:5]))
if __name__ == "__main__":
    arr = list(map(int, input().strip().split(' ')))
    miniMaxSum(arr)
"""

"""
You are in-charge of the cake for your niece's birthday and have decided the cake will have one candle for each year of her total age. When she blows out the candles, she’ll only be able to blow out the tallest ones.

For example, if your niece is turning  years old, and the cake will have  candles of height , , , , she will be able to blow out  candles successfully, since the tallest candle is of height  and there are  such candles.

Complete the function birthdayCakeCandles that takes your niece's age and an integer array containing height of each candle as input, and return the number of candles she can successfully blow out.

#!/bin/python3

import sys

def birthdayCakeCandles(n, ar):
    # Complete this function
    ars = sorted(ar)
    c = 0
    for x in ars:
        c += 1 if x == ars[(n-1)] else 0
    return(c)
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = birthdayCakeCandles(n, ar)
print(result)
"""

"""
Given a time in -hour AM/PM format, convert it to military (-hour) time.
Note: Midnight is  on a -hour clock, and  on a -hour clock. Noon is  on a -hour clock, and  on a -hour clock.
Input Format
A single string containing a time in -hour clock format (i.e.:  or ), where  and .

#!/bin/python3

import sys

def timeConversion(s):
    # Complete this function
    if s[8]=='A':
        if s[0:2]=='12':
            s='00'+s[2:8]
        else:
            s='{num:02d}'.format(num=int(s[0:2]))+s[2:8]
    else:
        if s[0:2]=='12':
            s='12'+s[2:8]
        else:
            s='{num:02d}'.format(num=(12+int(s[0:2])))+s[2:8]
    return(s)

s = input().strip()
result = timeConversion(s)
print(result)
"""

"""
Find the number of ways that a given integer, , can be expressed as the sum of the  powers of unique, natural numbers.

Input Format

The first line contains an integer . 
The second line contains an integer .

Constraints

Output Format
Output a single integer, the answer to the problem explained above.


#!/bin/python3

import sys

def powerSum(X, N):
    # Complete this function
    arr = []
    max = int(1+X**(1/float(N)))
    for i in range(1,max):
        if (i**N) <= X:
            arr.append((i**N))
    #print(arr)
    def recur(dig, ar):
        c = 0
        for i, j in enumerate(ar):
            if sum(dig) + j == X:
                c += 1
            else:
                c += recur(dig+[j], ar[i+1:])
        return c
    return recur([],arr)

if __name__ == "__main__":
    X = int(input().strip())
    N = int(input().strip())
    result = powerSum(X, N)
    print(result)
"""

"""

"""

###########################
###########################
# Stepic EXAM
"""
Вы анализируете эмоциональный окрас слов. На первой строчке подаётся числа n и m. В следующих n строках подаётся некоторое слово и его целочисленный окрас. Вам необходимо вывести m слов с максимальным окрасом (в любом порядке). Для некоторых слов может подаваться несколько разных значений окраса, надо выбирать максимальное.

Sample Input:

5 3
cool 0
no -10
apple 5
cool 10
school -20

Sample Output:

cool 10
apple 5
no -10
"""

n, m = map(int, input().split())
d = {}
for i in range(n):
    w, x = input().split()
    if w in d:
        if int(x) > d[w]:
            d[w] = int(x)
    else:
        d[w] = int(x)
c = 0
for i in sorted(d, key=d.__getitem__, reverse=True):
    if c < m:
        print(i, d[i])
        c += 1

"""
Напишите программу, на вход которой подаётся прямоугольная матрица в виде последовательности строк, заканчивающихся строкой, содержащей только строку "00".
Программа должна вывести матрицу того же размера, у которой каждый элемент в позиции i, j равен количеству различных элементов первой матрицы на позициях (i-1, j), (i+1, j), (i, j-1), (i, j+1). У крайних символов соседний элемент находится с противоположной стороны матрицы.
В случае одной строки/столбца элемент сам себе является соседом по соответствующему направлению.
Sample Input 1:
1 2 1
1 3 3
1 2 0
00
Sample Output 1:
2 3 4
2 3 3
3 4 3
Sample Input 2:
1
2
1
1
00
Sample Output 2:
2
2
2
1
"""
s = input()
b = []
c = []
while s != '00':
    b.append([int(i) for i in s.split()])
    c.append([int(i) for i in s.split()])
    s = input()
for i in range(len(b)):
    for j in range(len(b[i])):
        if (i == (len(b) - 1)) and (j != (len(b[i]) - 1)):
            c[i][j] = len(set([b[i - 1][j], b[0][j], b[i][j - 1], b[i][j + 1]]))
        elif (i != (len(b) - 1)) and (j == (len(b[i]) - 1)):
            c[i][j] = len(set([b[i - 1][j], b[i + 1][j], b[i][j - 1], b[i][0]]))
        elif (i == (len(b) - 1)) and (j == (len(b[i]) - 1)):
            c[i][j] = len(set([b[i - 1][j], b[0][j], b[i][j - 1], b[i][0]]))
        else:
            c[i][j] = len(set([b[i - 1][j], b[i + 1][j], b[i][j - 1], b[i][j + 1]]))
for i in c:
    for j in i:
        print(j, end=' ')
    print()

"""
Вам нужно написать класс ConcatReverse, который можно создать либо из строки
ConcatReverse('abcd')
либо из двух других объектов типа ConcatReverse:
a = ConcatReverse('abc')
b = ConcatReverse('def')
c = ConcatReverse(a, b)
Так класс должен иметь метод evaluate, который переворачивает строку в первом случае, а во втором случае склеивает результаты вызовов a.evaluate() и b.evaluate() и затем переворачивает результат.
Ваш код должен выглядеть следующим образом:
class ConcatReverse:
    def __init__(self, *args):
        # Создать объект
    def evaluate(self):
        # Вычислить строку
К примеру:
a = ConcatReverse('abc')
print(a.evaluate()) # 'cba'
b = ConcatReverse('def') => print(b.evaluate)=fed
c = ConcatReverse(a, b)
print(c.evaluate()) # 'defabc' => cbafed => defabc
print(a.evaluate()) # 'cba'
"""


class ConcatReverse:
    def __init__(self, *args):
        self.lst = []
        self.s = ''
        self.type = 0
        # print(type(args))
        # if not isinstance(args[0], ConcatReverse): #isinstance('a', str)
        if isinstance(args[0], str):
            self.s = args[0]
            self.type = 1
        else:
            self.type = 2
            for i in args:
                self.lst.append(i)

    def evaluate(self):
        # Вычислить строку
        self.ss = ''
        if self.type == 1:
            self.ss = self.s
            return self.ss[::-1]  # TypeError: 'JuryCR' object is not subscriptable
        # return ''.join(reversed(self.s)) #TypeError: argument to reversed() must be a sequence
        else:
            for i in range(len(self.lst)):
                self.ss += self.lst[i].evaluate()
            return self.ss[::-1]


"""
Реализуйте класс, представляющий собой структуру данных, которая позволяет осуществлять следующие операции с шарами из пространства ℤk.
add(r, *coords) – добавить точку
Принимает радиус r и k целых чисел, i-е из которых соответствует i-й координате центра шара.
remove(r, *coords) – удалить точку
Принимает то же, что и add. В случае, если шаров с такими радиусом и координатами несколько, удаляется только один из них. В случае, если соответствующих шаров нет, никакой шар не удаляется.
query(*coords) – запросить шары, покрывающие запрашиваемую точку
Принимает k целых чисел a1,…,ak. Шар (r,x) удовлетворяет запросу, если ∑ki=1(ai−xi)2≤r2, где xi - i
-ая координата центра шара. Результатом данной операции является объект-генератор. Последовательность, получаемая в результате итерации по генератору должна содержать ровно один раз каждый добавленный и не удаленный на момент запроса шар, удовлетворяющую запросу. Шары с одинаковыми координатами и радиусами считаются различными.

Формат решения:

class Spheres:
    def __init__(self, k):
        # k -- размерность пространства

    def add(self, r, *coords):
        # добавить шар

    def remove(self, r, *coords):
        # удалить шар

    def query(self, *coords):
        # запросить шары

Пример использования:

spheres = Spheres(2)
spheres.add(2, 1, 1)
spheres.add(1, -2, 1)
print(list(spheres.query(-1, 1))) # [(2, 1, 1), (1, -2, 1)]
print(list(spheres.query(0, 0))) # [(2, 1, 1)]
spheres.add(6, 0, -2)
spheres.remove(1, -2, 1)
print(list(spheres.query(-1, 1))) # [(2, 1, 1), (6, 0, -2)]
"""


class Spheres:
    def __init__(self, k):
        # k -- размерность пространства
        self.d = {}
        self.c = 0
        self.zk = k

    # for i in range(k):
    # self.d[i]=[]
    def add(self, r, *coords):
        # добавить шар
        # self.d[self.c]=[r,coords]
        self.d[self.c] = [r]
        self.d[self.c] += list(coords)
        self.c += 1

    def remove(self, r, *coords):
        # удалить шар
        def check(a, b):
            # print(a,b)
            if len(a) == len(b):
                for i in range(len(a)):
                    if a[i] == b[i]:
                        pass
                    else:
                        return False
                return True
            else:
                return False

        # print(r,coords)
        for x in self.d:
            # print(('remove',self.d[x][0],r,self.d[x][1:],list(coords)))
            if (self.d[x][0] == r) and check(self.d[x][1:], list(coords)):
                # print(x,self.d[x])
                # print(self.d.index(self.d[x]))
                self.d.pop(x)
                break

    def query(self, *coords):
        # запросить шары
        # print(11111, self.d)
        for x in self.d:
            self.s = 0
            # print(x)
            # print(self.d[x])
            # print(self.d[x][0],self.d[x][1])
            for j in range(self.zk):
                # print('a',coords[j],self.d[x][1][j],pow((coords[j] - self.d[x][1][j]),2))
                # self.s += pow((coords[j] - self.d[x][1][j]),2)
                self.s += pow((coords[j] - self.d[x][1:][j]), 2)
            # print('b',self.s,(self.d[x][0])^2)
            if self.s <= pow((self.d[x][0]), 2):
                yield self.d[x]


spheres = Spheres(2)
spheres.add(2, 1, 1)
spheres.add(1, -2, 1)
# print(spheres.d,spheres.c,spheres.zk)
print(list(spheres.query(-1, 1)))  # [(2, 1, 1), (1, -2, 1)]
print(list(spheres.query(0, 0)))  # [(2, 1, 1)]
spheres.add(6, 0, -2)
spheres.remove(1, -2, 1)
print(list(spheres.query(-1, 1)))  # [(2, 1, 1), (6, 0, -2)]

"""
Вам дана последовательность строк.
Выведите строки, содержащие в скобках текст, который не содержит внутри себя скобок.
Под текстом подразумевается последовательность символов, содержащая хотя бы один символ из группы \w.
Проверку требуемого условия реализуйте с помощью регулярного выражения.

Sample Input:

good (excellent) phrase
good (too bad) phrase
good ((recursive)) phrase
word () is not () in brackets
bad (() recursive) phrase
no brackets here

Sample Output:

good (excellent) phrase
good (too bad) phrase
good ((recursive)) phrase
"""

import sys
import re

for line in sys.stdin:
    line = line.rstrip()
    p = re.compile('\(\w+.(?!\(\))\w+.\)')
    if not (p.search(line) is None):
        print(line)

"""
Вам дана ссылка на HTML документ.
Посчитайте количество живых картинок в нем.

Живой картинкой назовем тег <img ... src="url" ... >, который отображается на странице, в котором url ведет на страницу, при запросе которой сервер вернет сообщение с status code равным 200 и заголовком Content-Type, начинающимся с image (например image/png)

Пример живой картинки
<img src="https://stepic.org/media/attachments/lesson/25669/nya.png">

Sample Input:
https://stepic.org/media/attachments/lesson/25669/sample.html

Sample Output:
2
"""
import requests
import re

c = 0
url = input()
# url='https://stepic.org/media/attachments/lesson/25669/sample.html'
res = requests.get(url)
if res.status_code == 200:
    # for url_img in re.findall('\<img.+?src=\"(.*)\".*?>',res.text): #\<img.+?src=\"(.*)\".*?>
    # for url_img in re.findall('\<img\s.*src=\"(.*)\">',res.text):#wr2
    # for url_img in re.findall('\<img\s.*src="([^"]+)">',res.text): #wr3\<img\s.*src=\"(.*)\">
    for url_img in re.findall('\<img[^>]*\ssrc="([^"]+)".*?>', res.text):
        res2 = requests.get(url_img)
        if res2.status_code == 200:  # & (not (re.search(r'^(image)',res2.headers['Content-Type']) is None)):
            if res2.headers['Content-Type'][0:6] == 'image/':
                c += 1
print(c)

"""
Два скучающих солдата играют в карточную войну. Их колода состоит ровно из n карт, пронумерованных различными числами от 1 до n. Исходно они делят между собой карты некоторым, возможно, не равным образом.

Правила игры следующие. На каждом ходу происходит сражение. Каждый игрок берет карту с вершины своей стопки и кладет на стол. Тот, у кого значение карты больше, выигрывает в этом сражении, берет обе карты со стола и кладет в низ своей стопки. Точнее говоря, сперва он берет карту противника и кладет в низ своей стопки, затем кладет свою карту в низ своей стопки. Если после какого-то хода стопка одного игрока становится пустой, то он проигрывает, а другой игрок побеждает.

Вам надо подсчитать, сколько будет сражений и кто победит, или сказать, что игра не прекратится.

Входные данные

В первой строке записано единственное целое число n (2 ≤ n ≤ 10) – количество карт.

Во второй строке записано целое число k1 (1 ≤ k1 ≤ n - 1) – количество карт у первого солдата. Затем следует k1 целых
чисел – значения карт первого солдата в порядке сверху вниз.

В третьей строке записано целое число k2 (k1 + k2 = n) – количество карт у второго солдата. Затем следует k2 целых
чисел – значения карт второго солдата сверху вниз.

Все значения карт различны.

Выходные данные

Если кто-то победит в этой игре, выведите 2 целых числа, где первое число обозначает количество сражений в игре, а второе равно 1, если побеждает первый солдат, и 2, если второй.

Если игра не закончится, а будет продолжаться вечно, выведите  -1.

Sample Input 1:
4
2 1 3
2 4 2

Sample Output 1:
6 2

Sample Input 2:
3
1 2
2 1 3

Sample Output 2:
-1
"""
n = int(input())
k1a = [i for i in input().split()]
k1 = k1a[0]
k1a.pop(0)
k2a = [i for i in input().split()]
k2 = k2a[0]
k2a.pop(0)
x = 0
while k1a and k2a:
    x += 1
    a, b = k1a.pop(0), k2a.pop(0)
    if int(a) > int(b):
        k1a += [b, a]
    else:
        k2a += [a, b]
    if x == 999999:
        print(-1)
        break
if x != 999999:
    if k1a:
        print(x, 1)
    else:
        print(x, 2)
"""
Подходящим назовем файл, имеющий расширение .csv с данными в формате CSV внутри.
Интересным назовем подходящий файл, в заголовке данных которого есть атрибут Pet в любом регистре (pet, PET, pEt, ...)
Хорошим назовем интересный файл, в данных которого строк с атрибутом Pet равным Cat меньше чем строк с атрибутом Pet равным Dog (слова pet, cat и dog могут быть в любом регистре).

Вам дана файловая структура. Найдите количество хороших файлов в ней.

Гарантируется, что любой файл с расширением .csv содержит данные в формате CSV.

Пример данных:
sample.zip

Пример ответа:
43

Основные данные:
data.zip

/home/truename/python/
/home/truename/python/sample/
/home/truename/python/data/
"""
import os
import csv

goodfile = 0
# path = "/home/truename/python/sample/"#==43
path = "/home/truename/python/data/"
csvfiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith((".csv"))]
for ff in csvfiles:
    with open(ff) as f:
        reader = csv.reader(f)
        r = next(reader)
        if "pet" in [x.lower() for x in r]:
            # print(ff)
            colmn = [x.lower() for x in r].index("pet")
            # print(r,colmn)
            dog = 0
            cat = 0
            for row in reader:
                if "dog" == row[colmn].lower():
                    dog += 1
                if "cat" == row[colmn].lower():
                    cat += 1
            if dog > cat:
                goodfile += 1
print(goodfile)
# dog>cat


###########################
###########################
# Stepic Basic and Apply
"""
Реализуйте программу, которая будет вычислять количество различных объектов в списке.
Два объекта a и b считаются различными, если a is b равно False.

Вашей программе доступна переменная с названием objects, которая ссылается на список, содержащий не более 100 объектов. Выведите количество различных объектов в этом списке.

Формат ожидаемой программы:

ans = 0
for obj in objects: # доступная переменная objects
    ans += 1

print(ans)


Примечание:
Количеством различных объектов называется максимальный размер множества объектов, в котором любые два объекта являются различными.

Рассмотрим пример:

objects = [1, 2, 1, 2, 3] # будем считать, что одинаковые числа соответствуют одинаковым объектам, а различные – различным


Тогда все различные объекты являют собой множество {1, 2, 3}﻿. Таким образом, количество различных объектов равно трём.
"""
ans = 0
b = []
for obj in objects:
    if obj not in b:
        b.append(obj)
        ans += 1
print(ans)
"""
Напишите реализацию функции closest_mod_5, принимающую в качестве единственного аргумента целое число x и возвращающую самое маленькое целое число y, такое что:

    y больше или равно x
    y делится нацело на 5

Формат того, что ожидается от вас в качестве ответа:

def closest_mod_5(x):
    if x % 5 == 0:
        return x
    return "I don't know :("
"""


def closest_mod_5(x):
    if x % 5 == 0:
        return x
    else:
        return closest_mod_5(x + 1)
    return "I don't know :("


"""
Сочетанием из n элементов по k называется подмножество этих n элементов размера k.
Два сочетания называются различными, если одно из сочетаний содержит элемент, который не содержит другое.
Числом сочетаний из n по k называется количество различных сочетаний из n по k. Обозначим это число за C(n, k).

Пример:
Пусть n = 3, т. е. есть три элемента (1, 2, 3). Пусть k = 2.
Все различные сочетания из 3 элементов по 2: (1, 2), (1, 3), (2, 3).
Различных сочетаний три, поэтому C(3, 2) = 3.

Несложно понять, что C(n, 0) = 1, так как из n элементов выбрать 0 можно единственным образом, а именно, ничего не выбрать.
Также несложно понять, что если k > n, то C(n, k) = 0, так как невозможно, например, из трех элементов выбрать пять.

Для вычисления C(n, k) в других случаях используется следующая рекуррентная формула: 
C(n, k) = C(n - 1, k) + C(n - 1, k - 1).

Реализуйте программу, которая для заданных n и k вычисляет C(n, k).

Вашей программе на вход подается строка, содержащая два целых числа n и k (1 ≤ n ≤ 10, 0 ≤ k ≤ 10).
Ваша программа должна вывести единственное число: C(n, k).

Примечание:
Считать два числа n и k﻿ вы можете, например, следующим образом:

n, k = map(int, input().split())

Sample Input 1:

3 2

Sample Output 1:

3


Sample Input 2:

10 5

Sample Output 2:

252
"""


def cNK(n, k):
    if k == 0:
        return 1
    else:
        if k > n:
            return 0
        else:
            return cNK(n - 1, k) + cNK(n - 1, k - 1)


n, k = map(int, input().split())
print(cNK(n, k))
"""
Реализуйте программу, которая будет эмулировать работу с пространствами имен. Необходимо реализовать поддержку создания пространств имен и добавление в них переменных.

В данной задаче у каждого пространства имен есть уникальный текстовый идентификатор – его имя.

Вашей программе на вход подаются следующие запросы:

    create <namespace> <parent> –  создать новое пространство имен с именем <namespace> внутри пространства <parent>
    add <namespace> <var> – добавить в пространство <namespace> переменную <var>
    get <namespace> <var> – получить имя пространства, из которого будет взята переменная <var> при запросе из пространства <namespace>, или None, если такого пространства не существует

Рассмотрим набор запросов

    add global a
    create foo global
    add foo b
    create bar foo
    add bar a

Структура пространств имен описанная данными запросами будет эквивалентна структуре пространств имен, созданной при выполнении данного кода

a = 0
def foo():
  b = 1
  def bar():
    a = 2

В основном теле программы мы объявляем переменную a, тем самым добавляя ее в пространство global. Далее мы объявляем функцию foo, что влечет за собой создание локального для нее пространства имен внутри пространства global. В нашем случае, это описывается командой create foo global. Далее мы объявляем внутри функции foo функцию bar, тем самым создавая пространство bar внутри пространства foo, и добавляем в bar переменную a.

Добавим запросы get к нашим запросам

    get foo a
    get foo c
    get bar a
    get bar b

Представим как это могло бы выглядеть в коде

a = 0
def foo():
  b = 1
  get(a)
  get(c)
  def bar():
    a = 2
    get(a)
    get(b)

 

Результатом запроса get будет имя пространства, из которого будет взята нужная переменная.
Например, результатом запроса get foo a будет global, потому что в пространстве foo не объявлена переменная a, но в пространстве global, внутри которого находится пространство foo, она объявлена. Аналогично, результатом запроса get bar b будет являться foo, а результатом работы get bar a будет являться bar.

Результатом get foo c будет являться None, потому что ни в пространстве foo, ни в его внешнем пространстве global не была объявлена переменная с.

Более формально, результатом работы get <namespace> <var> является

    <namespace>, если в пространстве <namespace> была объявлена переменная <var>
    get <parent> <var> – результат запроса к пространству, внутри которого было создано пространство <namespace>, если переменная не была объявлена
    None, если не существует <parent>, т. е. <namespace>﻿ – это global

Формат входных данных

В первой строке дано число n (1 ≤ n ≤ 100) – число запросов.
В каждой из следующих n строк дано по одному запросу.
Запросы выполняются в порядке, в котором они даны во входных данных.
Имена пространства имен и имена переменных представляют из себя строки длины не более 10, состоящие из строчных латинских букв.
Формат выходных данных

Для каждого запроса get выведите в отдельной строке его результат.


Sample Input:

9
add global a
create foo global
add foo b
get foo a
get foo c
create bar foo
add bar a
get bar a
get bar b

Sample Output:

global
None
bar
foo

"""


def getInst(inst, var):
    if inst == 'None':
        return None
    else:
        if var in s[inst]['vars']:
            return inst
        else:
            return getInst(s[inst]['pid'], var)


n = int(input())
s = {}
s['global'] = {'pid': 'None', 'vars': []}
# s['global']={}
# s['global'].update({'pid':'None'})
# s['global'].update({'vars':[]})
for i in range(n):
    cmd, a, b = input().split()
    if cmd == 'create':
        s[a] = {'pid': b, 'vars': []}
        # s[a]={}
        # s[a].update({'pid':b})
        # s[a].update({'vars':[]})
        # print(s)
    elif cmd == 'add':
        # s[a]['vars']+=b
        s[a]['vars'].append(b)
        # print(s)
    elif cmd == 'get':
        print(getInst(a, b))

"""


Реализуйте класс MoneyBox, для работы с виртуальной копилкой. 

Каждая копилка имеет ограниченную вместимость, которая выражается целым числом – количеством монет, которые можно положить в копилку. Класс должен поддерживать информацию о количестве монет в копилке, предоставлять возможность добавлять монеты в копилку и узнавать, можно ли добавить в копилку ещё какое-то количество монет, не превышая ее вместимость.

Класс должен иметь следующий вид

class MoneyBox:
    def __init__(self, capacity):
        # конструктор с аргументом – вместимость копилки

    def can_add(self, v):
        # True, если можно добавить v монет, False иначе

    def add(self, v):
        # положить v монет в копилку

При создании копилки, число монет в ней равно 0.

Примечание:
Гарантируется, что метод add(self, v) будет вызываться только если can_add(self, v) – True﻿.

"""


class MoneyBox:
    def __init__(self, capacity):
        # конструктор с аргументом – вместимость копилки
        self.capacity = capacity
        self.cnt = 0

    def can_add(self, v):
        # True, если можно добавить v монет, False иначе
        return v < self.capacity - self.cnt + 1

    def add(self, v):
        # положить v монет в копилку
        self.cnt += v


"""


Вам дается последовательность целых чисел и вам нужно ее обработать и вывести на экран сумму первой пятерки чисел из этой последовательности, затем сумму второй пятерки, и т. д.

Но последовательность не дается вам сразу целиком. С течением времени к вам поступают её последовательные части. Например, сначала первые три элемента, потом следующие шесть, потом следующие два и т. д.

Реализуйте класс Buffer, который будет накапливать в себе элементы последовательности и выводить сумму пятерок последовательных элементов по мере их накопления.

Одним из требований к классу является то, что он не должен хранить в себе больше элементов, чем ему действительно необходимо, т. е. он не должен хранить элементы, которые уже вошли в пятерку, для которой была выведена сумма.

Класс должен иметь следующий вид

class Buffer:
    def __init__(self):
        # конструктор без аргументов
    
    def add(self, *a):
        # добавить следующую часть последовательности

    def get_current_part(self):
        # вернуть сохраненные в текущий момент элементы последовательности в порядке, в котором они были     
        # добавлены


Пример работы с классом

buf = Buffer()
buf.add(1, 2, 3)
buf.get_current_part() # вернуть [1, 2, 3]
buf.add(4, 5, 6) # print(15) – вывод суммы первой пятерки элементов
buf.get_current_part() # вернуть [6]
buf.add(7, 8, 9, 10) # print(40) – вывод суммы второй пятерки элементов
buf.get_current_part() # вернуть []
buf.add(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) # print(5), print(5) – вывод сумм третьей и четвертой пятерки
buf.get_current_part() # вернуть [1]


Обратите внимание, что во время выполнения метода add﻿ выводить сумму пятерок может потребоваться несколько раз до тех пор, пока в буфере не останется менее пяти элементов.

"""


class Buffer:
    def __init__(self):
        # конструктор без аргументов
        self.lst = []

    def add(self, *a):
        # добавить следующую часть последовательности
        for i in a:
            self.lst.append(i)
            if len(self.lst) == 5:
                print(self.lst[0] + self.lst[1] + self.lst[2] + self.lst[3] + self.lst[4])
                self.lst = []

    def get_current_part(self):
        # вернуть сохраненные в текущий момент элементы последовательности в порядке, в котором они были     
        # добавлены
        return self.lst


"""
Вам дано описание наследования классов в следующем формате. 
<имя класса 1> : <имя класса 2> <имя класса 3> ... <имя класса k>
Это означает, что класс 1 отнаследован от класса 2, класса 3, и т. д.

Или эквивалентно записи:

class Class1(Class2, Class3 ... ClassK):
    pass

Класс A является прямым предком класса B, если B отнаследован от A:

class B(A):
    pass



Класс A является предком класса B, если 

    A = B;
    A - прямой предок B
    существует такой класс C, что C - прямой предок B и A - предок C


Например:

class B(A):
    pass

class C(B):
    pass

# A -- предок С



Вам необходимо отвечать на запросы, является ли один класс предком другого класса

Важное примечание:
Создавать классы не требуется.
Мы просим вас промоделировать этот процесс, и понять существует ли путь от одного класса до другого.
Формат входных данных

В первой строке входных данных содержится целое число n - число классов.

В следующих n строках содержится описание наследования классов. В i-й строке указано от каких классов наследуется i-й класс. Обратите внимание, что класс может ни от кого не наследоваться. Гарантируется, что класс не наследуется сам от себя (прямо или косвенно), что класс не наследуется явно от одного класса более одного раза.

В следующей строке содержится число q - количество запросов.

В следующих q строках содержится описание запросов в формате <имя класса 1> <имя класса 2>.
Имя класса – строка, состоящая из символов латинского алфавита, длины не более 50.
Формат выходных данных

Для каждого запроса выведите в отдельной строке слово "Yes", если класс 1 является предком класса 2, и "No", если не является. 

Sample Input:

4
A
B : A
C : A
D : B C
4
A B
B D
C D
D A

Sample Output:

Yes
Yes
Yes
No

"""


def getIsSubClass(c1, c2):
    b = False
    if c1 == c2:
        return True
    else:
        if len(s[c2]['pid']) == 0:
            return False
        else:
            if c1 in s[c2]['pid']:
                # b=b or True
                return True
            else:
                for j in s[c2]['pid']:
                    b = b or getIsSubClass(c1, j)
                return b


n = int(input())
s = {}
for i in range(n):
    x = [i for i in input().split()]
    if ":" in x:
        s[x[0]] = {'pid': x[2:]}
        for k in x[2:]:
            if k not in s:
                s[k] = {'pid': []}
    else:
        s[x[0]] = {'pid': []}
# print(s)
q = int(input())
for i in range(q):
    a, b = input().split()
    if getIsSubClass(a, b):
        print('Yes')
    else:
        print('No')
"""
Реализуйте структуру данных, представляющую собой расширенную структуру стек. Необходимо поддерживать добавление элемента на вершину стека, удаление с вершины стека, и необходимо поддерживать операции сложения, вычитания, умножения и целочисленного деления.

Операция сложения на стеке определяется следующим образом. Со стека снимается верхний элемент (top1), затем снимается следующий верхний элемент (top2), и затем как результат операции сложения на вершину стека кладется элемент, равный top1 + top2.

Аналогичным образом определяются операции вычитания (top1 - top2), умножения (top1 * top2) и целочисленного деления (top1 // top2).

Реализуйте эту структуру данных как класс ExtendedStack, отнаследовав его от стандартного класса list.
Требуемая структура класса:

class ExtendedStack(list):
    def sum(self):
        # операция сложения

    def sub(self):
        # операция вычитания

    def mul(self):
        # операция умножения

    def div(self):
        # операция целочисленного деления

 
Примечание
Для добавления элемента на стек используется метод append, а для снятия со стека – метод pop.
﻿Гарантируется, что операции будут совершаться только когда в стеке есть хотя бы два элемента.
"""


class ExtendedStack(list):
    def sum(self):
        # операция сложения
        self.append(self.pop() + self.pop())

    def sub(self):
        # операция вычитания
        self.append(self.pop() - self.pop())

    def mul(self):
        # операция умножения
        self.append(self.pop() * self.pop())

    def div(self):
        # операция целочисленного деления
        self.append(self.pop() // self.pop())


"""


Одно из применений множественного наследование – расширение функциональности класса каким-то заранее определенным способом. Например, если нам понадобится логировать какую-то информацию при обращении к методам класса.

Рассмотрим класс Loggable:

import time

class Loggable:
    def log(self, msg):
        print(str(time.ctime()) + ": " + str(msg))

У него есть ровно один метод log, который позволяет выводить в лог (в данном случае в stdout) какое-то сообщение, добавляя при этом текущее время.

Реализуйте класс LoggableList, отнаследовав его от классов list и Loggable таким образом, чтобы при добавлении элемента в список посредством метода append в лог отправлялось сообщение, состоящее из только что добавленного элемента.

Примечание
Ваша программа не должна содержать класс Loggable. При проверке вашей программе будет доступен этот класс, и он будет содержать метод log﻿, описанный выше.

"""


class LoggableList(list, Loggable):
    def append(self, l):
        super(LoggableList, self).append(l)
        self.log(l)


"""


Вашей программе будет доступна функция foo, которая может бросать исключения.

Вам необходимо написать код, который запускает эту функцию, затем ловит исключения ArithmeticError, AssertionError, ZeroDivisionError и выводит имя пойманного исключения.

Пример решения, которое вы должны отправить на проверку.

try:
    foo()
except Exception:
    print("Exception")
except BaseException:
    print("BaseException")

Подсказка: https://docs.python.org/3/library/exceptions.html#exception-hierarchy

"""
try:
    foo()
except ZeroDivisionError:
    print("ZeroDivisionError")
except ArithmeticError:
    print("ArithmeticError")
except AssertionError:
    print("AssertionError")
"""
Вам дано описание наследования классов исключений в следующем формате.
<имя исключения 1> : <имя исключения 2> <имя исключения 3> ... <имя исключения k>
Это означает, что исключение 1 наследуется от исключения 2, исключения 3, и т. д.

Или эквивалентно записи:

class Error1(Error2, Error3 ... ErrorK):
    pass


Антон написал код, который выглядит следующим образом.

try:
   foo()
except <имя 1>:
   print("<имя 1>")
except <имя 2>:
   print("<имя 2>")
...

Костя посмотрел на этот код и указал Антону на то, что некоторые исключения можно не ловить, так как ранее в коде будет пойман их предок. Но Антон не помнит какие исключения наследуются от каких. Помогите ему выйти из неловкого положения и напишите программу, которая будет определять обработку каких исключений можно удалить из кода.

Важное примечание:
В отличие от предыдущей задачи, типы исключений не созданы.
Создавать классы исключений также не требуется
Мы просим вас промоделировать этот процесс, и понять какие из исключений можно и не ловить, потому что мы уже ранее где-то поймали их предка.

Формат входных данных

В первой строке входных данных содержится целое число n - число классов исключений.

В следующих n строках содержится описание наследования классов. В i-й строке указано от каких классов наследуется i-й класс. Обратите внимание, что класс может ни от кого не наследоваться. Гарантируется, что класс не наследуется сам от себя (прямо или косвенно), что класс не наследуется явно от одного класса более одного раза.

В следующей строке содержится число m - количество обрабатываемых исключений.
Следующие m строк содержат имена исключений в том порядке, в каком они были написаны у Антона в коде.
Гарантируется, что никакое исключение не обрабатывается дважды.
Формат выходных данных

Выведите в отдельной строке имя каждого исключения, обработку которого можно удалить из кода, не изменив при этом поведение программы. Имена следует выводить в том же порядке, в котором они идут во входных данных.
Пример теста 1

Рассмотрим код

try:
   foo()
except ZeroDivision :
   print("ZeroDivision")
except OSError:
   print("OSError")
except ArithmeticError:
   print("ArithmeticError")
except FileNotFoundError:
   print("FileNotFoundError")


...



По условию этого теста, Костя посмотрел на этот код, и сказал Антону, что исключение FileNotFoundError можно не ловить, ведь мы уже ловим OSError -- предок FileNotFoundError

Sample Input:

4
ArithmeticError
ZeroDivisionError : ArithmeticError
OSError
FileNotFoundError : OSError
4
ZeroDivisionError
OSError
ArithmeticError
FileNotFoundError

Sample Output:

FileNotFoundError
"""


def getIsSubClass(c1, lst):
    b = False
    if c1 in lst:
        return True
    else:
        # print(lst)
        if len(s[c1]['pid']) == 0:
            return False
        else:
            for c2 in s[c1]['pid']:
                if c2 in lst:
                    return True
                else:
                    b = b or getIsSubClass(c2, lst)
            return b


n = int(input())
s = {}
for i in range(n):
    x = [i for i in input().split()]
    if ":" in x:
        s[x[0]] = {'pid': x[2:]}
        for k in x[2:]:
            if k not in s:
                s[k] = {'pid': []}
    else:
        s[x[0]] = {'pid': []}
# print(s)
q = int(input())
l = []
for i in range(q):
    a = input()
    if getIsSubClass(a, l):
        print(a)
    l.append(a)
"""
Реализуйте класс PositiveList, отнаследовав его от класса list, для хранения положительных целых чисел.
Также реализуйте новое исключение NonPositiveError.

В классе PositiveList переопределите метод append(self, x) таким образом, чтобы при попытке добавить неположительное целое число бросалось исключение NonPositiveError и число не добавлялось, а при попытке добавить положительное целое число, число добавлялось бы как в стандартный list.

В данной задаче гарантируется, что в качестве аргумента x метода append всегда будет передаваться целое число.

Примечание:
Положительными считаются числа, строго большие ﻿нуля.
"""


class NonPositiveError(Exception):
    pass


class PositiveList(list):
    def append(self, x):
        if x <= 0:
            raise NonPositiveError
        else:
            super(PositiveList, self).append(x)


"""
В первой строке дано три числа, соответствующие некоторой дате date -- год, месяц и день.
Во второй строке дано одно число days -- число дней.

Вычислите и выведите год, месяц и день даты, которая наступит, когда с момента исходной даты date пройдет число дней, равное days.

Примечание:
Для решения этой задачи используйте стандартный модуль datetime.
Вам будут полезны класс datetime.date для хранения даты и класс datetime.timedelta﻿ для прибавления дней к дате.

Sample Input 1:

2016 4 20
14

Sample Output 1:

2016 5 4


Sample Input 2:

2016 2 20
9

Sample Output 2:

2016 2 29


Sample Input 3:

2015 12 31
1

Sample Output 3:

2016 1 1
"""
import datetime

y, m, d = [int(i) for i in input().split()]
dcnt = int(input())
date_ = (datetime.date(y, m, d) + datetime.timedelta(days=dcnt))
print(date_.year, date_.month, date_.day)
"""
Алиса владеет интересной информацией, которую хочет заполучить Боб.
Алиса умна, поэтому она хранит свою информацию в зашифрованном файле.
У Алисы плохая память, поэтому она хранит все свои пароли в открытом виде в текстовом файле.

Бобу удалось завладеть зашифрованным файлом с интересной информацией и файлом с паролями, но он не смог понять какой из паролей ему нужен. Помогите ему решить эту проблему.

Алиса зашифровала свою информацию с помощью библиотеки simple-crypt.
Она представила информацию в виде строки, и затем записала в бинарный файл результат работы метода simplecrypt.encrypt.

Вам необходимо установить библиотеку simple-crypt, и с помощью метода simplecrypt.decrypt узнать, какой из паролей служит ключом для расшифровки файла с интересной информацией.

Ответом для данной задачи служит расшифрованная интересная информация Алисы.

Файл с информацией
Файл с паролями

Примечание:
Для того, чтобы считать все данные из бинарного файла, можно использовать, например, следующий код:

with open("encrypted.bin", "rb") as inp:
    encrypted = inp.read()


Примечание:
﻿Работа с файлами рассмотрена в следующем уроке, поэтому вы можете вернуться к этой задаче после просмотра следующего урока.
"""
"""
Одним из самых часто используемых классов в Python является класс filter. Он принимает в конструкторе два аргумента a и f – последовательность и функцию, и позволяет проитерироваться только по таким элементам x из последовательности a, что f(x) равно True. Будем говорить, что в этом случае функция f допускает элемент x, а элемент x является допущенным.

В данной задаче мы просим вас реализовать класс multifilter, который будет выполнять ту же функцию, что и стандартный класс filter, но будет использовать не одну функцию, а несколько. 

Решение о допуске элемента будет приниматься на основании того, сколько функций допускают этот элемент, и сколько не допускают. Обозначим эти количества за pos и neg.

Введем понятие решающей функции – это функция, которая принимает два аргумента – количества pos и neg, и возвращает True, если элемент допущен, и False иначе.

Рассмотрим процесс допуска подробнее на следующем примере.
a = [1, 2, 3]
f2(x) = x % 2 == 0 # возвращает True, если x делится на 2
f3(x) = x % 3 == 0
judge_any(pos, neg) = pos >= 1 # возвращает True, если хотя бы одна функция допускает элемент

В этом примере мы хотим отфильтровать последовательность a и оставить только те элементы, которые делятся на два или на три.

Функция f2 допускает только элементы, делящиеся на два, а функция f3 допускает только элементы, делящиеся на три. Решающая функция допускает элемент в случае, если он был допущен хотя бы одной из функций f2 или f3, то есть элементы, которые делятся на два или на три.

Возьмем первый элемент x = 1.
f2(x) равно False, т. е. функция f2 не допускает элемент x.
f3(x) также равно False, т. е. функция f3 также не допускает элемент x.
В этом случае pos = 0, так как ни одна функция не допускает x, и соответственно neg = 2.
judge_any(0, 2) равно False, значит мы не допускаем элемент x = 1.

Возьмем второй элемент x = 2.
f2(x) равно True
f3(x) равно False
pos = 1, neg = 1
judge_any(1, 1) равно True, значит мы допускаем элемент x = 2.

Аналогично для третьего элемента x = 3.

Таким образом, получили последовательность допущенных элементов [2, 3].

Класс должен обладать следующей структурой:

class multifilter:
    def judge_half(pos, neg):
        # допускает элемент, если его допускает хотя бы половина фукнций (pos >= neg)

    def judge_any(pos, neg):
        # допускает элемент, если его допускает хотя бы одна функция (pos >= 1)

    def judge_all(pos, neg):
        # допускает элемент, если его допускают все функции (neg == 0)

    def __init__(self, iterable, *funcs, judge=judge_any):
        # iterable - исходная последовательность
        # funcs - допускающие функции
        # judge - решающая функция

    def __iter__(self):
        # возвращает итератор по результирующей последовательности


Пример использования:
﻿

def mul2(x):
    return x % 2 == 0

def mul3(x):
    return x % 3 == 0

def mul5(x):
    return x % 5 == 0


a = [i for i in range(31)] # [0, 1, 2, ... , 30]

print(list(multifilter(a, mul2, mul3, mul5))) 
# [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30]

print(list(multifilter(a, mul2, mul3, mul5, judge=multifilter.judge_half))) 
# [0, 6, 10, 12, 15, 18, 20, 24, 30]

print(list(multifilter(a, mul2, mul3, mul5, judge=multifilter.judge_all))) 
# [0, 30]
"""


class multifilter:
    def judge_half(pos, neg):
        # допускает элемент, если его допускает хотя бы половина фукнций (pos >= neg)
        return pos >= neg

    def judge_any(pos, neg):
        # допускает элемент, если его допускает хотя бы одна функция (pos >= 1)
        return pos >= 1

    def judge_all(pos, neg):
        # допускает элемент, если его допускают все функции (neg == 0)
        return neg == 0

    def __init__(self, iterable, *funcs, judge=judge_any):
        # iterable - исходная последовательность
        # funcs - допускающие функции
        # judge - решающая функция
        self.iterable = iterable
        self.judge = judge
        self.funcs = funcs
        # pos=len([x for x in iterable for f in funcs if f(x)])

    def __iter__(self):
        # возвращает итератор по результирующей последовательности
        for x in self.iterable:
            pos = 0
            neg = 0
            for f in self.funcs:
                if f(x):
                    pos += 1
                else:
                    neg += 1
            if self.judge(pos, neg):
                yield x


"""
Целое положительное число называется простым, если оно имеет ровно два различных делителя, то есть делится только на единицу и на само себя.
Например, число 2 является простым, так как делится только на 1 и 2. Также простыми являются, например, числа 3, 5, 31, и еще бесконечно много чисел.
Число 4, например, не является простым, так как имеет три делителя – 1, 2, 4. Также простым не является число 1, так как оно имеет ровно один делитель – 1.

Реализуйте функцию-генератор primes, которая будет генерировать простые числа в порядке возрастания, начиная с числа 2.

Пример использования:﻿

print(list(itertools.takewhile(lambda x : x <= 31, primes())))
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
"""


def is_prime(a):
    if a <= 1:
        return False
    if a % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(a)) + 1, 1):
        if a % i == 0:
            return False
    return True


def primes():
    x = 2
    yield 2
    while True:
        x += 1
        if is_prime(x):
            yield x


"""
Вашей программе на вход подаются три строки s, a, b, состоящие из строчных латинских букв.
За одну операцию вы можете заменить все вхождения строки a в строку s на строку b.

Например, s = "abab", a = "ab", b = "ba", тогда после выполнения одной операции строка s перейдет в строку "baba", после выполнения двух и операций – в строку "bbaa", и дальнейшие операции не будут изменять строку s﻿.

Необходимо узнать, после какого минимального количества операций в строке s не останется вхождений строки a, либо же определить, что это невозможно.

Выведите одно число – минимальное число операций, после применения которых в строке s не останется вхождений строки a.
Если после применения любого числа операций в строке s останутся вхождения строки a, выведите Impossible.

Sample Input 1:

ababa
a
b

Sample Output 1:

1


Sample Input 2:

ababa
b
a

Sample Output 2:

1


Sample Input 3:

ababa
c
c

Sample Output 3:

0


Sample Input 4:

ababac
c
c

Sample Output 4:

Impossible
"""
s, a, b = [input().strip() for _ in range(3)]
if a in s:
    if a in b:
        print("Impossible")
    else:
        cnt = 0
        while a in s:
            cnt += 1
            s = s.replace(a, b)
        print(cnt)
else:
    print(0)
"""
Вашей программе на вход подаются две строки s и t, состоящие из строчных латинских букв.

Выведите одно число – количество вхождений строки t в строку s.

Пример:
s = "abababa"
t = "aba"

Вхождения строки t в строку s:
abababa
abababa
abababa﻿

Sample Input 1:

abababa
aba

Sample Output 1:

3


Sample Input 2:

abababa
abc

Sample Output 2:

0


Sample Input 3:

abc
abc

Sample Output 3:

1


Sample Input 4:

aaaaa
a

Sample Output 4:

5

"""
s, t = [input().strip() for _ in range(2)]
cnt = 0
i = 0
while i < len(s):
    if s.find(t, i) > -1:
        # print(s.find(t,i))
        if s.find(t, i) > i:
            i = s.find(t, i) + 1
        else:
            i += 1
        cnt += 1
    else:
        i += 1
print(cnt)
"""
Вам дана последовательность строк.
Выведите строки, содержащие "cat" в качестве подстроки хотя бы два раза.

Примечание:
Считать все строки по одной из стандартного потока ввода вы можете, например, так

import sys

for line in sys.stdin:
    line = line.rstrip()
    # process line


Sample Input:

catcat
cat and cat
catac
cat
ccaatt

Sample Output:

catcat
cat and cat
"""
import sys
import re

for line in sys.stdin:
    line = line.rstrip()
    # p = re.compile('(cat(.*)){2,}')
    # if not p.match(line) is None:
    #	print(line)
    p = re.compile('(cat(.*)){2,}')
    # print(p.search(line))
    if not (p.search(line) is None):
        print(line)
"""
Вам дана последовательность строк.
Выведите строки, содержащие "cat" в качестве слова.

Примечание:
Для работы со словами используйте группы символов \b и \B.
Описание этих групп вы можете найти в документации.

Sample Input:

cat
catapult and cat
catcat
concat
Cat
"cat"
!cat?

Sample Output:

cat
catapult and cat
"cat"
!cat?
"""
import sys
import re

for line in sys.stdin:
    line = line.rstrip()
    p = re.compile(r'\bcat\b')
    if not (p.search(line) is None):
        print(line)
"""
Вам дана последовательность строк.
Выведите строки, содержащие две буквы "z﻿", между которыми ровно три символа.

Sample Input:

zabcz
zzz
zzxzz
zz
zxz
zzxzxxz

Sample Output:

zabcz
zzxzz
"""
import sys
import re

for line in sys.stdin:
    line = line.rstrip()
    if not (re.search(r'z(.{3})z', line) is None):
        print(line)
"""
Вам дана последовательность строк.
Выведите строки, содержащие обратный слеш "\﻿".

Sample Input:

\w denotes word character
No slashes here

Sample Output:

\w denotes word character
"""
import sys
import re

for line in sys.stdin:
    line = line.rstrip()
    if not (re.search(r'.*\\.*', line) is None):
        print(line)
"""
Вам дана последовательность строк.
Выведите строки, содержащие слово, состоящее из двух одинаковых частей (тандемный повтор).

Sample Input:

blabla is a tandem repetition
123123 is good too
go go
aaa

Sample Output:

blabla is a tandem repetition
123123 is good too
"""
import sys
import re

for line in sys.stdin:
    line = line.rstrip()
    if not (re.search(r'\b([a-zA-Z0-9]+)\1\b', line) is None):
        print(line)
"""
Вам дана последовательность строк.
В каждой строке замените все вхождения подстроки "human" на подстроку "computer"﻿ и выведите полученные строки.

Sample Input:

I need to understand the human mind
humanity

Sample Output:

I need to understand the computer mind
computerity
"""
import sys
import re

for line in sys.stdin:
    line = line.rstrip()
    print(re.sub('(human)', 'computer', line))
"""
Вам дана последовательность строк.
В каждой строке замените первое вхождение слова, состоящего только из латинских букв "a" (регистр не важен), на слово "argh".

Примечание:
Обратите внимание на параметр count у функции sub﻿.

Sample Input:

There’ll be no more "Aaaaaaaaaaaaaaa"
AaAaAaA AaAaAaA

Sample Output:

There’ll be no more "argh"
argh AaAaAaA
"""
import sys, re

for line in sys.stdin:
    line = line.rstrip()
    print(re.sub(r'\b([aA]+)\b', 'argh', line, count=1))
"""
Вам дана последовательность строк.
В каждой строке поменяйте местами две первых буквы в каждом слове, состоящем хотя бы из двух букв.
Буквой считается символ из группы \w﻿.

Sample Input:

this is a text
"this' !is. ?n1ce,

Sample Output:

htis si a etxt
"htis' !si. ?1nce,
"""
import sys, re


def repl(m):
    return m.group(2) + m.group(1) + m.group(3)


for line in sys.stdin:
    line = line.rstrip()
    print(re.sub(r'\b(\w)(\w)(\w*)\b', repl, line))
"""
Вам дана последовательность строк.
В каждой строке замените все вхождения нескольких одинаковых букв на одну букву.
Буквой считается символ из группы \w.

Sample Input:

attraction
buzzzz

Sample Output:

atraction
buz
"""
import sys, re


def repl(m):
    return m.group(1)


for line in sys.stdin:
    line = line.rstrip()
    print(re.sub(r'(\w)\1+', repl, line))
"""
Примечание:
Эта задача является дополнительной, то есть ее решение не принесет вам баллы.
Задача сложнее остальных задач из этого раздела, и идея ее решения выходит за рамки простого понимания регулярных выражений как средства задания шаблона строки.
Мы решили включить данную задачу в урок, чтобы показать, что регулярным выражением можно проверить не только "внешний вид" строки, но и заложенный в ней смысл.


Вам дана последовательность строк.
Выведите строки, содержащие двоичную запись числа, кратного 3.

Двоичной записью числа называется его запись в двоичной системе счисления.

Примечание 2:
﻿Данная задача очень просто может быть решена приведением строки к целому числу и проверке остатка от деления на три, но мы все же предлагаем вам решить ее, не используя приведение к числу.

Sample Input:

0
10010
00101
01001
Not a number
1 1
0 0

Sample Output:

0
10010
01001
"""
import sys, re


def d3(t):
    multiplier = 1
    accumulator = 0
    for bit in t:
        accumulator = (accumulator + int(bit) * multiplier) % 3
        multiplier = 3 - multiplier
    return accumulator == 0


# print(re.findall(r'\d','0101111'))
for line in sys.stdin:
    line = line.rstrip()
    if not (re.search(r'^([01]+)\Z', line) is None):
        # print(line)
        if d3(line):
            print(line)
"""
Рассмотрим два HTML-документа A и B.
Из A можно перейти в B за один переход, если в A есть ссылка на B, т. е. внутри A есть тег <a href="B">, возможно с дополнительными параметрами внутри тега.
Из A можно перейти в B за два перехода если существует такой документ C, что из A в C можно перейти за один переход и из C в B можно перейти за один переход.

Вашей программе на вход подаются две строки, содержащие url двух документов A и B.
Выведите Yes, если из A в B можно перейти за два перехода, иначе выведите No.

Обратите внимание на то, что не все ссылки внутри HTML документа могут вести на существующие HTML документы.

Sample Input 1:

https://stepic.org/media/attachments/lesson/24472/sample0.html
https://stepic.org/media/attachments/lesson/24472/sample2.html

Sample Output 1:

Yes


Sample Input 2:

https://stepic.org/media/attachments/lesson/24472/sample0.html
https://stepic.org/media/attachments/lesson/24472/sample1.html

Sample Output 2:

No


Sample Input 3:

https://stepic.org/media/attachments/lesson/24472/sample1.html
https://stepic.org/media/attachments/lesson/24472/sample2.html

Sample Output 3:

Yes
"""


def checkWayBy2Step(url_b, url_e):
    res = requests.get(url_b)
    if res.status_code == 200:
        # print(res.text)
        for url2 in re.findall('\<a\s.*href=\"(.*)\">', res.text):
            # print(url2)
            res2 = requests.get(url2)
            if res2.status_code == 200:
                for url3 in re.findall('\<a\s.*href=\"(.*)\">', res2.text):
                    if url3 == url_e:
                        return True
    return False


url1 = input()
url2 = input()
if checkWayBy2Step(url1, url2):
    print('Yes')
else:
    print('No')
"""
Вашей программе на вход подается ссылка на HTML файл.
Вам необходимо скачать этот файл, затем найти в нем все ссылки вида <a ... href="..." ... > и вывести список сайтов, на которые есть ссылка.

Сайтом в данной задаче будем называть имя домена вместе с именами поддоменов. То есть, это последовательность символов, которая следует сразу после символов протокола, если он есть, до символов порта или пути, если они есть, за исключением случаев с относительными ссылками вида
<a href="../some_path/index.html">﻿.

Сайты следует выводить в алфавитном порядке.

Пример HTML файла:

<a href="http://stepic.org/courses">
<a href='https://stepic.org'>
<a href='http://neerc.ifmo.ru:1345'>
<a href="ftp://mail.ru/distib" >
<a href="ya.ru">
<a href="www.ya.ru">
<a href="../skip_relative_links">

Пример ответа:

mail.ru
neerc.ifmo.ru
stepic.org
www.ya.ru
ya.ru
"""
import requests
import re


def url_path_to_dict(path):
    pattern = (r'^'
               r'((?P<schema>.+?)://)?'
               r'((?P<user>.+?)(:(?P<password>.*?))?@)?'
               r'(?P<host>.*?)'
               r'(:(?P<port>\d+?))?'
               r'(?P<path>/.*?)?'
               r'(?P<query>[?].*?)?'
               r'$'
               )
    regex = re.compile(pattern)
    m = regex.match(path)
    d = m.groupdict() if m is not None else None

    return d


d = []
ur = input()
res = requests.get(ur)
if res.status_code == 200:
    for url in re.findall('\<a\s.*href=[\"\'](.*?)[\"\'\s]+?', res.text):
        if (re.search(r'^\.\.', url) is None):
            u = url_path_to_dict(url)
            if not u['host'] is None:
                if not u['host'] in d:
                    d.append(u['host'])
else:
    print(res.status_code)
for i in sorted(d):
    print(i)
"""
Вам дано описание наследования классов в формате JSON.
Описание представляет из себя массив JSON-объектов, которые соответствуют классам. У каждого JSON-объекта есть поле name, которое содержит имя класса, и поле parents, которое содержит список имен прямых предков.

Пример:
[{"name": "A", "parents": []}, {"name": "B", "parents": ["A", "C"]}, {"name": "C", "parents": ["A"]}]

﻿Эквивалент на Python:

class A:
    pass

class B(A, C):
    pass

class C(A):
    pass


Гарантируется, что никакой класс не наследуется от себя явно или косвенно, и что никакой класс не наследуется явно от одного класса более одного раза.

Для каждого класса вычислите предком скольких классов он является и выведите эту информацию в следующем формате.

<имя класса> : <количество потомков>

Выводить классы следует в лексикографическом порядке.

Sample Input:

[{"name": "A", "parents": []}, {"name": "B", "parents": ["A", "C"]}, {"name": "C", "parents": ["A"]}]

Sample Output:

A : 3
B : 1
C : 2

"""
import json


def getIsSubClass(c1, c2):
    b = False
    # if c1==c2:
    #	return True
    # else:
    if len(dd[c2]['pid']) == 0:
        return False
    else:
        if c1 in dd[c2]['pid']:
            return True
        else:
            for j in dd[c2]['pid']:
                b = b or getIsSubClass(c1, j)
            return b


d = json.loads(input())
dd = {}
for i in d:
    dd[i['name']] = {'pid': i['parents'], 'cnt': 0}
for i in dd:
    for j in dd:
        if getIsSubClass(i, j):
            dd[i]['cnt'] += 1

for i in sorted(dd):
    print(i, ':', dd[i]['cnt'] + 1)
"""
В этой задаче вам необходимо воспользоваться API сайта numbersapi.com

Вам дается набор чисел. Для каждого из чисел необходимо узнать, существует ли интересный математический факт об этом числе.

Для каждого числа выведите Interesting, если для числа существует интересный факт, и Boring иначе.
Выводите информацию об интересности чисел в таком же порядке, в каком следуют числа во входном файле.

Пример запроса к интересному числу:
http://numbersapi.com/31/math?json=true

Пример запроса к скучному числу:
http://numbersapi.com/999/math?json=true

Пример входного файла:
31
999
1024
502

﻿Пример выходного файла:
Interesting
Boring
Interesting
Boring
"""
"""
В этой задаче вам необходимо воспользоваться API сайта artsy.net

API проекта Artsy предоставляет информацию о некоторых деятелях искусства, их работах, выставках.

В рамках данной задачи вам понадобятся сведения о деятелях искусства (назовем их, условно, художники).

Вам даны идентификаторы художников в базе Artsy.
Для каждого идентификатора получите информацию о имени художника и годе рождения.
Выведите имена художников в порядке неубывания года рождения. В случае если у художников одинаковый год рождения, выведите их имена в лексикографическом порядке.

Работа с API Artsy

Полностью открытое и свободное API предоставляют совсем немногие проекты. В большинстве случаев, для получения доступа к API необходимо зарегистрироваться в проекте, создать свое приложение, и получить уникальный ключ (или токен), и в дальнейшем все запросы к API осуществляются при помощи этого ключа.

Чтобы начать работу с API проекта Artsy, вам необходимо пройти на стартовую страницу документации к API https://developers.artsy.net/start и выполнить необходимые шаги, а именно зарегистрироваться, создать приложение, и получить пару идентификаторов Client Id и Client Secret. Не публикуйте эти идентификаторы.

После этого необходимо получить токен доступа к API. На стартовой странице документации есть примеры того, как можно выполнить запрос и как выглядит ответ сервера. Мы приведем пример запроса на Python.

import requests
import json

client_id = '...'
client_secret = '...'

# инициируем запрос на получение токена
r = requests.post("https://api.artsy.net/api/tokens/xapp_token",
                  data={
                      "client_id": client_id,
                      "client_secret": client_secret
                  })

# разбираем ответ сервера
j = json.loads(r.text)

# достаем токен
token = j["token"]

 

Теперь все готово для получения информации о художниках. На стартовой странице документации есть пример того, как осуществляется запрос и как выглядит ответ сервера. Пример запроса на Python.

# создаем заголовок, содержащий наш токен
headers = {"X-Xapp-Token" : token}


# инициируем запрос с заголовком
r = requests.get("https://api.artsy.net/api/artists/4d8b92b34eb68a1b2c0003f4", headers=headers)

# разбираем ответ сервера
j = json.loads(r.text)


Примечание:
﻿В качестве имени художника используется параметр sortable_name в кодировке UTF-8.

Пример входных данных:
4d8b92b34eb68a1b2c0003f4
537def3c139b21353f0006a6
4e2ed576477cc70001006f99

Пример выходных данных:
Abbott Mary
Warhol Andy
Abbas Hamra

Примечание для пользователей Windows
При открытии файла для записи на Windows по умолчанию используется кодировка CP1251, в то время как для записи имен на сайте используется кодировка UTF-8, что может привести к ошибке при попытке записать в файл имя с необычными символами. Вы можете использовать print, или аргумент encoding функции open.
"""
"""
Вам дано описание пирамиды из кубиков в формате XML.
Кубики могут быть трех цветов: красный (red), зеленый (green) и синий (blue﻿).
Для каждого кубика известны его цвет, и известны кубики, расположенные прямо под ним.

Пример:

<cube color="blue">
  <cube color="red">
    <cube color="green">
    </cube>
  </cube>
  <cube color="red">
  </cube>
</cube>

 

Введем понятие ценности для кубиков. Самый верхний кубик, соответствующий корню XML документа имеет ценность 1. Кубики, расположенные прямо под ним, имеют ценность 2. Кубики, расположенные прямо под нижележащими кубиками, имеют ценность 3. И т. д.

Ценность цвета равна сумме ценностей всех кубиков этого цвета.

Выведите через пробел три числа: ценности красного, зеленого и синего цветов.

Sample Input:

<cube color="blue"><cube color="red"><cube color="green"></cube></cube><cube color="red"></cube></cube>

Sample Output:

4 3 1
"""
import xml.etree.ElementTree as etree


def gatherdict(r, lev):
    # print('gatherdict',lev)
    if not r.attrib['color'] in d:
        d[r.attrib['color']] = lev
    else:
        d[r.attrib['color']] += lev
    for c in r:  # .findall('.//cube'):
        # print(c.attrib)
        if len(c.attrib['color']) > 0:
            # print(c.attrib['color'],lev + 1)
            # if not c.attrib['color'] in d:
            #	d[c.attrib['color']] = lev + 1
            # else:
            #	d[c.attrib['color']] += lev + 1
            gatherdict(c, lev + 1)


# else:
#	d[r.attrib['color']] = lev


# print(d)
d = {}
n = input()
# n='<cube color="blue"><cube color="red"><cube color="green"></cube></cube><cube color="red"></cube></cube>'
# tree = etree.parse(n)
# root = tree.getroot()
root = etree.fromstring(n)
# print(root.attrib)
# for child in root:
#	print(child.attrib)
# print(root.findall('.//cube'))
gatherdict(root, 1)
s = ''
if 'red' in d:
    s = str(d['red'])
else:
    s = '0'
if 'green' in d:
    s += ' ' + str(d['green'])
else:
    s += ' 0'
if 'blue' in d:
    s += ' ' + str(d['blue'])
else:
    s += ' 0'
print(s)

###########################
###########################
# Stepic Adaptive Python
n = int(input())
a = [int(i) for i in input().split()]
m = int(input())
b = [int(i) for i in input().split()]
c = []
for i in range(m):
    k = 0
    d = b[i] - a[0]
    for j in range(n):
        if abs(b[i] - a[j]) < d:
            k = j
            d = abs(b[i] - a[j])
    c.append(k)
for i in range(m):
    print(c[i], end=' ')

###########################
###########################


###########################
###########################


###########################
###########################


###########################
###########################


###########################
###########################
