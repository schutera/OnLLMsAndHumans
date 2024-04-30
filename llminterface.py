import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import re


def clean_prompt(prompt):
    # Remove punctuation marks
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt)
    # Replace spaces with underscores
    cleaned_prompt = cleaned_prompt.replace(' ', '_')
    return cleaned_prompt


class Falcon7B:
    def __init__(self):
        # Load 7B model and tokenizer
        self.model_name = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def generate_response(self, prompt, temperature):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Explicitly set attention_mask and pad_token_id
        attention_mask = torch.ones_like(input_ids)

        output = self.model.generate(
            input_ids,
            max_length=150,
            temperature=temperature,
            num_beams=5,
            do_sample=True, # the model is configured to use sampling-based generation (where the model samples from its output distribution to generate text)
            no_repeat_ngram_size=2,
            attention_mask=attention_mask,  # Set attention mask
            pad_token_id=self.tokenizer.eos_token_id  # Set pad token id to eos token id
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


if __name__ == "__main__":

    llm = Falcon7B()

    # Define the prompts and temperatures
    prompts = ["Pick a number between 1 and 100.",
               "Pick a random number between 1 and 100."]
               # "Pick a number between 0 and 9.",
               # "Pick a random number between 0 and 9."]
    temperatures = [1.0, 0.8, 0.6, 0.4, 0.2]

    # Loop over each combination of prompt and temperature
    for prompt in prompts:
        for temperature in temperatures:
            print(f"Prompt: '{prompt}', Temperature: {temperature}")

            # Array to store answers
            answers = []

            # Loop 1000 times
            for i in range(1000):
                print(f"{i + 1} / 1000")
                # Answer the question based on the prompt and temperature
                result = llm.generate_response(prompt, temperature)
                # Append the answer to the array
                answers.append(result)
                print(result)

            # Save the answers to a CSV file named after the prompt and temperature
            csv_file = f"{clean_prompt(prompt)}_t{temperature}.csv"
            with open(csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Answer"])  # Write header
                for answer in answers:
                    writer.writerow([answer[-1]])

            print(f"Answers saved to '{csv_file}'")



