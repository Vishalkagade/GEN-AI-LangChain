from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from typing import Optional,Literal,TypedDict,Annotated

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: Annotated[str, Field(description="Name of the person")]
    age: Annotated[int, Field(gt=18,description="Age of the person")]
    city: Annotated[str, Field(description="City of the person")]

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Extract the name, age and city from the fictional {input} person \n" \
    "{format_instructions}",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
final_result = chain.invoke({"input" : "Germany"})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
print(final_result)