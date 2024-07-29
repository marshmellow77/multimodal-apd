import asyncio
import os
import pandas as pd
from vertexai.generative_models import GenerativeModel, Part, HarmBlockThreshold, HarmCategory
import re
import aiofiles
from tqdm.asyncio import tqdm
import datetime
from dotenv import load_dotenv
import vertexai
import time
import aioconsole
from tqdm.asyncio import tqdm_asyncio
from colorama import Fore, Style


# Initialize colorama
from colorama import init
init(autoreset=True)


class MultiModalAPD:
    def __init__(self, num_prompts, starting_prompt, df_test, metaprompt_template_path, generation_model_name, generation_config, safety_settings, eval_model_name, eval_config, review_model_name, review_config, use_system_prompt, enable_user_feedback, show_thought_process):
        self.num_prompts = num_prompts
        self.starting_prompt = starting_prompt
        self.df_test = df_test
        self.metaprompt_template_path = metaprompt_template_path
        self.generation_model_name = generation_model_name
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.eval_model_name = eval_model_name
        self.eval_config = eval_config
        self.review_model_name = review_model_name  # New parameter for review model
        self.review_config = review_config  # New parameter for review model config
        self.enable_user_feedback = enable_user_feedback
        self.show_thought_process = show_thought_process

        self.use_system_prompt = use_system_prompt
        self.generation_model = GenerativeModel(self.generation_model_name)
        self.eval_model = GenerativeModel(self.eval_model_name)
        self.review_model = GenerativeModel(self.review_model_name)  # Initialize review model
        self.user_feedback = ""

        # Create the "runs" folder if it doesn't exist
        self.runs_folder = "runs"
        os.makedirs(self.runs_folder, exist_ok=True)
        
        self.run_folder = self.create_run_folder()
        self.results_file = os.path.join(self.run_folder, 'results.txt')
        self.results_static_file = os.path.join(self.run_folder, 'results_static.txt')
        self.image_results_file = os.path.join(self.run_folder, 'image_results.txt')
        
        self.df_test['review_response'] = None  # Add a new column for review responses
        self.df_test['extracted_number'] = None  # Initialize extracted_number column
        self.df_test['is_correct'] = None  # Initialize is_correct column


    def create_run_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_type = "system_prompt" if self.use_system_prompt else "content_prompt"
        run_folder = os.path.join(self.runs_folder, f'run_{timestamp}_{folder_type}')  # Join with runs_folder
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def create_prompt_subfolder(self, prompt_number):
        prompt_folder = os.path.join(self.run_folder, f'prompt_{prompt_number}')
        os.makedirs(prompt_folder, exist_ok=True)
        return prompt_folder

    async def get_user_feedback(self, timeout=120):
        try:
            if self.user_feedback:
                print("\nCurrent Feedback:")
                print(Fore.RED + self.user_feedback)

            return await asyncio.wait_for(self.get_user_input(), timeout=timeout)
        except asyncio.TimeoutError:
            print("No feedback received within timeout. Resuming...")
            return self.user_feedback 
        
    async def get_user_input(self):  # Separate function to get user input
        while True:
            new_feedback = await aioconsole.ainput("Enter your feedback for the model (or type 'skip' to skip): ")
            if new_feedback.lower() == "skip":
                return self.user_feedback  # Keep existing feedback or empty string

            if hasattr(self, 'user_feedback'):
                choice = await aioconsole.ainput("Replace (r) or append (a) previous feedback? (r/a): ")
                if choice.lower() == 'r':
                    self.user_feedback = new_feedback
                elif choice.lower() == 'a':
                    self.user_feedback += "\n" + new_feedback
                else:
                    await aioconsole.aprint("Invalid choice. Please enter 'r' or 'a'.")
            else:
                self.user_feedback = new_feedback

            confirm = await aioconsole.ainput("Is this feedback correct? (y/n): ")
            if confirm.lower() == 'y':
                return self.user_feedback


    def read_and_sort_prompt_accuracies(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        pattern = re.compile(r'<PROMPT>\n<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: ([0-9.]+)\n</ACCURACY>\n</PROMPT>', re.DOTALL)
        matches = pattern.findall(content)
        
        sorted_prompts = sorted(matches, key=lambda x: float(x[1]))  # Sort in ascending order
        return sorted_prompts

    def write_sorted_prompt_accuracies(self, file_path, sorted_prompts):
        with open(file_path, 'w') as f:
            for prompt, accuracy in sorted_prompts:
                f.write(f"<PROMPT>\n<PROMPT_TEXT>\n{prompt}\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: {accuracy}\n</ACCURACY>\n</PROMPT>\n\n")

    def update_metaprompt(self, file_path, metaprompt_template_path):
        sorted_prompts = self.read_and_sort_prompt_accuracies(file_path)
        self.write_sorted_prompt_accuracies(file_path, sorted_prompts)
        
        prompt_scores = "\n".join([f"Prompt: {prompt}\nAccuracy: {accuracy}" for prompt, accuracy in sorted_prompts])
        
        with open(metaprompt_template_path, 'r') as f:
            metaprompt_template = f.read()
        
        metaprompt = metaprompt_template.format(prompt_scores=prompt_scores, human_feedback=self.user_feedback)
        
        return metaprompt

    async def generate_and_extract(self, row, prompt, semaphore):
        async with semaphore:
            try:
                image = Part.from_data(
                    mime_type="image/jpeg",
                    data=row.decoded_image["bytes"]
                )
                if self.use_system_prompt:
                    content = [image]
                    eval_model = GenerativeModel(
                        self.eval_model_name,
                        generation_config=self.eval_config,
                        safety_settings=self.safety_settings,
                        system_instruction=prompt
                    )
                else:
                    content = [prompt, image]
                    eval_model = GenerativeModel(
                        self.eval_model_name,
                        generation_config=self.eval_config,
                        safety_settings=self.safety_settings
                    )

                start_time = time.time()
                response = await eval_model.generate_content_async(
                    content,
                    stream=False,
                )
                end_time = time.time()
                duration = end_time - start_time
                model_response = response.text
                token_usage = response.usage_metadata.total_token_count
                
                review_prompt = (
                    "You are a review model tasked with evaluating whether two transcripts are absolutely identical. Not only with respect to the text, but also who said it. "
                    "Please determine if the final answer provided in the response is correct based on the ground truth. "
                    "If the model response says that location is 'right' that is equivalent to 'Me' in the ground truth "
                    "If the model response says that location is 'left' that is equivalent to 'Other' in the ground truth "
                    "Other than that be very strict. "
                    "Respond with 'True' if the final answer is correct and 'False' if it is not. "
                    "Only respond with 'True' or 'False', nothing else.\n\n"
                    "Model Response:\n{model_response}\n\n"
                    "Ground Truth:\n{ground_truth}"
                ).format(model_response=model_response, ground_truth=row['answer'])


                # Now use the review model to compare the model response with the ground truth
                review_response = await self.review_model.generate_content_async(
                    [review_prompt],
                    generation_config=self.review_config,
                    safety_settings=self.safety_settings,
                    stream=False,
                )
                is_correct = review_response.text.strip().lower() == 'true'  # Check if the response is 'True'
                
                pid = row['pid']  # Assuming 'pid' is the identifier in your DataFrame
                self.df_test.loc[self.df_test['pid'] == pid, 'model_response'] = model_response
                self.df_test.loc[self.df_test['pid'] == pid, 'review_response'] = review_response.text.strip()  # Store the review response
                self.df_test.loc[self.df_test['pid'] == pid, 'is_correct'] = int(is_correct)


                return model_response, duration, token_usage, is_correct 
            except Exception as e:
                # print(e)
                return None, None, None, False

    async def evaluate_prompt(self, prompt, prompt_number):
        semaphore = asyncio.Semaphore(100)
        tasks = [self.generate_and_extract(row, prompt, semaphore) for _, row in self.df_test.iterrows()]
        
        # Create a tqdm progress bar
        with tqdm_asyncio(total=len(tasks), desc=f"Prompt {prompt_number} Evaluation") as pbar:

            async def wrapped_task(task):
                result = await task
                pbar.update(1)  # Update progress bar after task completion
                return result

            # Run tasks with progress bar updates
            results = await asyncio.gather(*[wrapped_task(task) for task in tasks])

        # Update the DataFrame with model responses and extracted numbers
        for i, (model_response, duration, token_usage, is_correct) in enumerate(results):
            pid = self.df_test.iloc[i]['pid']  # Get the 'pid' from the filtered DataFrame
            self.df_test.loc[self.df_test['pid'] == pid, 'model_response'] = model_response
            self.df_test.loc[self.df_test['pid'] == pid, 'duration'] = duration  # Save the duration
            self.df_test.loc[self.df_test['pid'] == pid, 'token_usage'] = token_usage  # Save the token usage

        overall_accuracy = sum(is_correct for _, _, _, is_correct in results) / len(results)

        # Create a subfolder for the prompt
        prompt_folder = self.create_prompt_subfolder(prompt_number)

        # Save the prompt in a text file within the subfolder
        prompt_file_path = os.path.join(prompt_folder, 'prompt.txt')
        with open(prompt_file_path, 'w') as f:
            f.write(prompt)

        # Save the evaluation results in a CSV file within the subfolder
        csv_file_path = os.path.join(prompt_folder, 'evaluation_results.csv')
        self.df_test[["pid", "answer", "model_response", "review_response", "is_correct", "duration", "token_usage"]].to_csv(csv_file_path, index=False)

        return overall_accuracy

    async def main(self):
        global trigger_feedback

        prompt_accuracies = []

        for i in range(self.num_prompts+1):
            await aioconsole.aprint("="*150)
            await aioconsole.aprint(Fore.BLUE + f"Prompt number {i}")

            # Get feedback every 5 iterations
            if i > 0 and i < self.num_prompts and i % 10 == 0 and self.enable_user_feedback:
                # self.user_feedback = self.get_user_feedback()
                self.user_feedback = await self.get_user_feedback()
            
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
                
                if self.show_thought_process:
                    await aioconsole.aprint("-"*150)
                    await aioconsole.aprint(Fore.YELLOW + response.text)
                    await aioconsole.aprint("-"*150)
                
                match = re.search(r'\<(.*?)\>', response.text, re.DOTALL)
                if match:
                    new_prompt = match.group(1)
                else:
                    await aioconsole.aprint(Fore.RED + "No new prompt found")
                    continue
            
            # Evaluate new prompt across all images in the DataFrame
            accuracy = await self.evaluate_prompt(new_prompt, i)
            prompt_accuracies.append((new_prompt, accuracy))
            await aioconsole.aprint("-"*150)
            await aioconsole.aprint(Fore.GREEN + f"Overall accuracy for prompt: {accuracy:.3f}")
            await aioconsole.aprint("="*150)
            
            # Append to results.txt
            async with aiofiles.open(self.results_file, 'a') as f:
                await f.write(f"<PROMPT>\n<PROMPT_TEXT>\n{new_prompt}\n</PROMPT_TEXT>\n<ACCURACY>\nAccuracy: {accuracy:.3f}\n</ACCURACY>\n</PROMPT>\n\n")
            
            # Append to results_static.txt with prompt number
            async with aiofiles.open(self.results_static_file, 'a') as f:
                await f.write(f"\nPrompt number: {i}\nPrompt: {new_prompt}\nAccuracy: {accuracy:.3f}\n\n")
                await f.write("="*150)            
            # Read, sort, and write the updated prompt accuracies to results.txt
            sorted_prompts = self.read_and_sort_prompt_accuracies(self.results_file)
            self.write_sorted_prompt_accuracies(self.results_file, sorted_prompts)
            
            # Wait for 60 seconds to avoid exceeding API quota, except after the last run
            if i < self.num_prompts - 1:
                await aioconsole.aprint("Waiting for 1 seconds to avoid exceeding API quota...")
                async for _ in tqdm_asyncio(range(1), desc="Waiting", unit="s"):
                    await asyncio.sleep(1)

               
if __name__ == "__main__":
    load_dotenv()
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    vertexai.init(project=project_id, location=location)

    num_prompts = 500
    starting_prompt = """You are analyzing an image of a text conversation. Your job is to carefully extract each message and figure out who sent it. Look at which side of the image the message bubble is on: left or right. Crucially, the same color bubble always means the same sender. If the color changes, it's a different person talking. Ignore things like reactions or extra marks on the messages, but do include any emojis. Give me the results in JSON format. Each message should be an object with "text" for what it says and "location" as either "left" or "right"."""
    
    # Load your DataFrame
    df_test = pd.read_pickle('chat_identification/apd-chat.pkl')  # pickle ensures data type preservation

    metaprompt_template_path = 'chat_identification/metaprompt_template_chat.txt'
    generation_model_name = "gemini-1.5-pro"
    generation_config = {
        "temperature": 0.7,
    }
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
    eval_model_name = "gemini-1.5-pro"
    eval_config = {
        "temperature": 0, "max_output_tokens": 1000
    }
    review_model_name = "gemini-1.5-flash"  # Use the same model or a different one for review
    review_config = {
        "temperature": 0, "max_output_tokens": 10  # Adjust as needed for the review model
    }
    use_system_prompt = False
    enable_user_feedback = True
    show_thought_process = True

    apd = MultiModalAPD(
        num_prompts, starting_prompt, df_test, 
        metaprompt_template_path, generation_model_name, generation_config, safety_settings, 
        eval_model_name, eval_config, review_model_name, review_config, use_system_prompt, enable_user_feedback, show_thought_process
    )

    asyncio.run(apd.main())
