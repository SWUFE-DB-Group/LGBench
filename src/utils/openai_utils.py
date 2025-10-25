from openai import OpenAI


def get_llm_output(base_url, api_key, model, user_prompt, sys_prompt=None, ):
    client = OpenAI(base_url=base_url,
                    api_key=api_key)
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    if sys_prompt:
        messages.insert(0, {"role": "system", "content": sys_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content
