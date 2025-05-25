from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name : str = "Vishal"
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt = 0, lt = 10, default = 5, description = "A decimal value representing the cgpa of the student")


new_student = {"age" : 27, "email": "kagadevishal@gmail.com", "cgpa": 8}

student = Student(**new_student)

student_json = student.model_dump_json()
student_dict = student.model_dump()
print(type(student_dict))