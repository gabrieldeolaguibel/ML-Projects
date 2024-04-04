import base64
from openai import OpenAI

#open_ai_key = ""
client = OpenAI(api_key=open_ai_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    


#path to image
image_path = "/Users/gabrieldeolaguibel/IE/DevOps_Assignement1/Statistical-Learning-and-Prediction/class_hackathon/parte_amistoso_9_39.jpg"

# getting the base64 encoded image
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model = 'gpt-4-1106-vision-preview',
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'From this image in spanish, extract any other useful information you can read and tell me about what you can observe'},
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpg;base64,{base64_image}' ,
				
                    }
                }
            ]
        }
    ],
    #max_tokens = 500
)

print('Completion Tokens', response.usage.completion_tokens)
print('Prompt Tokens', response.usage.prompt_tokens)
print('Total Tokens', response.usage.total_tokens)

# extract: names dates address country phone

#print the response
print(response.choices[0].message)
print(response.choices[0].message.content)