from dotenv import load_dotenv

load_dotenv()

import dheera_ai

from autoevals.llm import *

###################

# dheera_ai completion call
question = "which country has the highest population"
response = dheera_ai.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": question}],
)
print(response)
# use the auto eval Factuality() evaluator

print("calling evaluator")
evaluator = Factuality()
result = evaluator(
    output=response.choices[0]["message"][
        "content"
    ],  # response from dheera_ai.completion()
    expected="India",  # expected output
    input=question,  # question passed to dheera_ai.completion
)

print(result)
