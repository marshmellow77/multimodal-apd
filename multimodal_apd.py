import asyncio
import os
from vertexai.generative_models import GenerativeModel, Part, Content, Image
import vertexai.preview.generative_models as generative_models
import re
import aiofiles
from tqdm.asyncio import tqdm
import datetime
from dotenv import load_dotenv
import vertexai



class MultiModalAPD:
    def __init__(self, total_tasks, num_prompts, starting_prompt, images_and_ground_truths, metaprompt_template_path, generation_model_name, generation_config, safety_settings, eval_model_name, eval_config):
        self.total_tasks = total_tasks
        self.num_prompts = num_prompts
        self.starting_prompt = starting_prompt
        self.images_and_ground_truths = images_and_ground_truths
        self.metaprompt_template_path = metaprompt_template_path
        self.generation_model_name = generation_model_name
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.eval_model_name = eval_model_name
        self.eval_config = eval_config
        self.run_folder = self.create_run_folder()
        self.results_file = os.path.join(self.run_folder, 'results.txt')
        self.image_results_file = os.path.join(self.run_folder, 'image_results.txt')
        self.generation_model = GenerativeModel(self.generation_model_name)
        self.eval_model = GenerativeModel(self.eval_model_name)

    def create_run_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f'run_{timestamp}'
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def read_and_sort_prompt_accuracies(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        pattern = re.compile(r'Prompt: (.*?)\nAccuracy: ([0-9.]+)', re.DOTALL)
        matches = pattern.findall(content)
        
        sorted_prompts = sorted(matches, key=lambda x: float(x[1]))  # Sort in ascending order
        return sorted_prompts

    def write_sorted_prompt_accuracies(self, file_path, sorted_prompts):
        with open(file_path, 'w') as f:
            for prompt, accuracy in sorted_prompts:
                f.write(f"Prompt: {prompt}\nAccuracy: {accuracy}\n\n")

    def update_metaprompt(self, file_path, metaprompt_template_path):
        sorted_prompts = self.read_and_sort_prompt_accuracies(file_path)
        self.write_sorted_prompt_accuracies(file_path, sorted_prompts)
        
        prompt_scores = "\n".join([f"Prompt: {prompt}\nAccuracy: {accuracy}" for prompt, accuracy in sorted_prompts])
        
        with open(metaprompt_template_path, 'r') as f:
            metaprompt_template = f.read()
        
        metaprompt = metaprompt_template.format(prompt_scores=prompt_scores)
        
        return metaprompt

    def extract_number_from_last_line(self, text):
        lines = text.strip().split('\n')
        last_line = lines[-1]
        pattern = re.compile(r'(\d+(\.\d+)?√?\d*)')
        match = pattern.search(last_line)
        if match:
            return match.group()
        else:
            return None

    async def generate_and_extract(self, task_id, content, ground_truth):
        try:
            response = await self.eval_model.generate_content_async(
                        content,
                        generation_config=self.eval_config,
                        safety_settings=self.safety_settings,
                        stream=False,
                    )
            result = self.extract_number_from_last_line(response.text)
            return result == ground_truth
        except:
            return False

    async def write_image_accuracy(self, image_path, prompt, accuracy):
        async with aiofiles.open(self.image_results_file, 'a') as f:
            await f.write(f"Image: {image_path}\nPrompt: {prompt}\nAccuracy: {accuracy}\n\n")

    async def evaluate_prompt(self, prompt):
        if self.total_tasks > 100:
            raise ValueError("total_tasks should not be larger than 100")
        
        overall_results = []
        for image_info in self.images_and_ground_truths:
            image_path = image_info["image_path"]
            ground_truth = image_info["ground_truth"]
            
            image = Image.load_from_file(image_path)
            content = ["Problem:", image, prompt]
            
            print(f"Processing {self.total_tasks} tasks for image {image_path}")
            
            tasks = [self.generate_and_extract(i, content, ground_truth) for i in range(self.total_tasks)]
            image_results = await asyncio.gather(*tasks)
            
            image_accuracy = sum(image_results) / len(image_results)
            await self.write_image_accuracy(image_path, prompt, image_accuracy)  # Write accuracy per image
            overall_results.extend(image_results)
        
        overall_accuracy = sum(overall_results) / len(overall_results)
        return overall_accuracy

    async def main(self):
        prompt_accuracies = []

        for i in range(self.num_prompts):
            print(f"Prompt number {i+1}")
            
            if i == 0:
                new_prompt = self.starting_prompt
            else:
                metaprompt = self.update_metaprompt(self.results_file, self.metaprompt_template_path)
                
                response = self.generation_model.generate_content(
                    metaprompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    stream=False,
                )
                
                match = re.search(r'\[(.*?)\]', response.text, re.DOTALL)
                if match:
                    new_prompt = match.group(1)
                else:
                    print("No match found")
                    continue
            
            # Evaluate new prompt across all images
            accuracy = await self.evaluate_prompt(new_prompt)
            prompt_accuracies.append((new_prompt, accuracy))
            print(f"Overall accuracy for prompt: {accuracy}")
            
            # Append to file
            async with aiofiles.open(self.results_file, 'a') as f:
                await f.write(f"Prompt: {new_prompt}\nAccuracy: {accuracy}\n\n")
            
            # Read, sort, and write the updated prompt accuracies
            sorted_prompts = self.read_and_sort_prompt_accuracies(self.results_file)
            self.write_sorted_prompt_accuracies(self.results_file, sorted_prompts)
            
            # Wait for 60 seconds to avoid exceeding API quota, except after the last run
            if i < self.num_prompts - 1:
                print("Waiting for 60 seconds to avoid exceeding API quota...")
                for _ in tqdm(range(60), desc="Waiting", unit="s"):
                    await asyncio.sleep(1)
                
if __name__ == "__main__":
    load_dotenv()
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    vertexai.init(project=project_id, location=location)

    total_tasks = 100
    num_prompts = 2
    starting_prompt = "Solve this problem."
    images_and_ground_truths = [
        {"image_path": "2.jpg", "ground_truth": "36"},
        {"image_path": "3.png", "ground_truth": "12"},
        # Add more images and their ground truths here
    ]

    metaprompt_template_path = 'metaprompt_template.txt'
    generation_model_name = "gemini-1.5-pro"
    generation_config = {
        "temperature": 0.7,
    }
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    }
    eval_model_name = "gemini-1.5-flash"
    eval_config = {
        "temperature": 0,
    }

    apd = MultiModalAPD(
        total_tasks, num_prompts, starting_prompt, images_and_ground_truths, 
        metaprompt_template_path, generation_model_name, generation_config, safety_settings, 
        eval_model_name, eval_config
    )

    asyncio.run(apd.main())