# decorator to json
import functools


def to_json(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import json
        return json.JSONEncoder().encode(func(*args, **kwargs))

    return wrapper


"""
@to_json
def get_data():
    return {
        'data': 42
    }
print(get_data())  # вернёт '{"data": 42}'
"""
