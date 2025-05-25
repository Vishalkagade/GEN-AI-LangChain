from typing import TypedDict, List, Union


class Person(TypedDict):

    name: str
    age: int

new_person : Person = {"name": "Vishal", "age": 27}
print(type(new_person["name"]))