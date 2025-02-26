import openai
from openai import OpenAI

from groq import Groq


def call_openai(messages, max_tokens, temperature, model="gpt-4o-mini-2024-07-18"):

    client = OpenAI(api_key="xxxxxxxxxxx")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    return response.choices[0].message.content.strip()


def llama3_1_8B(messages, max_tokens, temperature, model="llama-3.1-8b-instant"):
    client = Groq(api_key="xxxxxxxxxxx")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stream=True,
        stop=None,
    )
    # print(completion)

    result =  ''

    for chunk in completion:
        # print('---------------------------------------')
        # print(chunk.choices[0].delta.content or "", end="")
        result += chunk.choices[0].delta.content or ""

    return result


# test
if __name__ == '__main__':
    from prompt_rep import __zero_shot_prompt__, __zero_shot_head__

    end_date = "2022-12-31"
    h = 7
    title = "The title of the article"
    text = "Summary and category of the article"

    instant_prompt = __zero_shot_prompt__.format(h=h, end_date=end_date, title=title, text=text)
    messages=[
            {"role": "system", "content": __zero_shot_head__},
            {"role": "user", "content": instant_prompt}
        ]
    
    result = llama3_1_8B(messages, max_tokens=100, temperature=0.5)
    print('-------------------00000--------------------')
    print(result)