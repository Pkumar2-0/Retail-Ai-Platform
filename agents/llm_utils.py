from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_response(prompt):

    prompt = f"""
    Answer this properly:

    {prompt}

    Response:
    """

    result = generator(
        prompt,

        max_new_tokens=60,

        do_sample=True,

        temperature=0.6,

        top_k=50,

        top_p=0.95,

        repetition_penalty=1.2,

        pad_token_id=50256
    )

    generated_text = result[0]["generated_text"]

    response = generated_text.split(
        "Response:"
    )[-1].strip()

    return response