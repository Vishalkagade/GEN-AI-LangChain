from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

# Models
model_chatgpt = ChatOpenAI()
model_hf = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="conversational"
    )
)

# Prompts
prompt1 = PromptTemplate(
    template="Create a summary of the performance review of the employee: {review}",
    input_variables=["review"]
)
prompt2 = PromptTemplate(
    template="Create 3 improvement areas for the employee from the following review: {review}",
    input_variables=["review"]
)
prompt3 = PromptTemplate(
    template="Merge the summary and improvement areas into a single report for the employee.\n\nSummary: {summary}\n\nImprovement Areas: {improvement_areas}",
    input_variables=["summary", "improvement_areas"]
)

# Output parser
parser = StrOutputParser()

# Chains
chain1 = prompt1 | model_chatgpt | parser
chain2 = prompt2 | model_hf | parser

# Run chains in parallel
parallel_chain = RunnableParallel({
    "summary": chain1,
    "improvement_areas": chain2
})

# Merge outputs
merge_chain = prompt3 | model_chatgpt | parser
final_chain = parallel_chain | merge_chain

# Input text
review_text = """During the review period, John Doe was responsible for developing and maintaining backend APIs, collaborating closely with the front-end and DevOps teams, ensuring high-quality code with appropriate test coverage, and participating in regular sprint planning and code reviews. His role is crucial to the delivery of several core features within our web application infrastructure.

John consistently delivered on his responsibilities with a high level of professionalism and skill. He met nearly all of his delivery deadlines, with 90% of his assigned tasks completed on time, demonstrating strong time management and accountability. The quality of his code was exceptional, averaging only 2.5 bugs per 1,000 lines of code, which is well within the acceptable benchmark. Additionally, John maintained a high standard of unit testing, with an average test coverage of 88%, surpassing our 85% target.

Peer feedback from code reviews was overwhelmingly positive, with very few rejections or requested changes during the review process. John is also recognized for his strong communication skills and collaborative attitude. Among his most notable achievements this quarter, he led the development of a new billing module, optimized several core APIs to improve response time by 30%, and took the initiative to mentor two new interns, helping them onboard effectively. These efforts significantly contributed to the team's overall productivity and morale.

One area for improvement is John's responsiveness during cross-team meetings. On a few occasions, delays in clarifying implementation details with the front-end team led to minor bottlenecks. Enhancing timely communication in such settings would help prevent misunderstandings and improve coordination across departments.

John's manager, Jane Smith, noted that he has demonstrated excellent technical capability and attention to detail. She believes that John is ready to take on additional responsibilities and encourages him to be more proactive during interdepartmental discussions.

Overall, John's performance for this quarter is rated as Exceeds Expectations with an overall score of 4.6 out of 5. He has expressed satisfaction with his current role and appreciates the learning opportunities. For the next review period, John aims to lead the backend architecture revamp for the reporting dashboard, conduct monthly knowledge-sharing sessions, and improve his sprint delivery rate to 95% or higher."""

# Run the chain
result = final_chain.invoke({
    "review": review_text
})

print(result)

final_chain.get_graph().draw_ascii()