import asyncio
from vertexai.generative_models import GenerativeModel, Part, Content, Image
import vertexai.preview.generative_models as generative_models
import re


prompt = """You are an AI Math Tutor, tasked with explaining a math problem depicted in an image to a student. Your explanation should be incorporated directly onto the image itself using annotations, diagrams, and clear markings. Each step of the solution should be visually demonstrated and accompanied by a concise, easy-to-understand explanation written directly on the image.  Avoid using any textual explanations outside of the image itself. Your goal is to make the solution clear and understandable solely through visual aids and on-image annotations."""


model = GenerativeModel("gemini-1.5-flash")

generation_config = {
    "temperature": 0,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
}


image = Image.load_from_file("3.png")

content = ["Problem:", image, prompt]

def extract_number_from_last_line(text):
    # Split text into lines
    lines = text.strip().split('\n')
    # Get the last line
    last_line = lines[-1]
    
    # Regular expression to match numbers, optionally with a square root sign (√)
    pattern = re.compile(r'(\d+(\.\d+)?√?\d*)')
    
    # Search for the pattern in the last line
    match = pattern.search(last_line)
    
    if match:
        return match.group()
    else:
        return None

async def generate_and_extract(i, semaphore):
    async with semaphore:
        print(f"Starting task {i}")
        response = await model.generate_content_async(
                    content,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )
        try:
            result = extract_number_from_last_line(response.text)
            if i == 50:
                print(response.text)
            # print(f"Task {i} result: {result}")
            return result == "12"
        except:
            return False

async def main():
    total_tasks = 100
    batch_size = 100
    semaphore = asyncio.Semaphore(100)

    results = []
    for batch_start in range(0, total_tasks, batch_size):
        batch_end = min(batch_start + batch_size, total_tasks)
        print(f"Processing batch {batch_start // batch_size + 1} from {batch_start} to {batch_end - 1}")
        
        tasks = [generate_and_extract(i, semaphore) for i in range(batch_start, batch_end)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        if batch_end < total_tasks:
            print(f"Batch {batch_start // batch_size + 1} completed. Sleeping for 60 seconds...")
            await asyncio.sleep(60)

            
    print(f"Accuracy: {sum(results)/len(results)}")

if __name__ == "__main__":
    asyncio.run(main())