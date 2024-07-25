from datasets import load_dataset
from vertexai.generative_models import GenerativeModel, HarmBlockThreshold, HarmCategory, Part
import ast

generation_config = {
    "temperature": 0,
}

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

system_prompt = """You are a math teacher and you want to test your students in trigonometry.
Analyse a bunch of images and tell me which are trigonometric problems.
At the end of your response, provide a list with only the image numbers in it in square brackets."""

model = GenerativeModel("gemini-1.5-flash",
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        system_instruction=system_prompt)

dataset = load_dataset("AI4Math/MathVista", split='testmini')
df = dataset.to_pandas()

trigonometry_images = []

for i in range(10):
    print(i)
    content = []
    
    for j in range(i*100, (i+1)*100):
        content.append(f"Image {j}: ")
        image = Part.from_data(
                mime_type="image/jpeg",
                data=df.iloc[j].decoded_image["bytes"]
            )
        content.append(image)
        content.append("\n")

    response = model.generate_content(content)
    
    # print(response.text)
    
    last_line = response.text.splitlines()[-1]
    l = ast.literal_eval(last_line)
    
    trigonometry_images.extend(l)

df_trigonometry = df.iloc[trigonometry_images]
# Regular expression to match numerical values
regex = r'^[+-]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?$|^√\d+$|^\d+°$|^±?\d+(\.\d+)?$'

# Filter the DataFrame
df_trigonometry_numbers = df_trigonometry[df_trigonometry['answer'].str.match(regex)]

assert(len(df_trigonometry_numbers) >= 100)

df_test = df_trigonometry_numbers.sample(n=100, random_state=42)
df_test.to_pickle("apd-testdata.pkl") # for data type preservation
df_test.to_csv("apd-testdata.csv", index=False) # for easier manual inspection
