import argparse
import json
import logging
import os
import re
import time

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


load_dotenv()

SYSTEM_MESSAGE ="Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"


def list_model_generation_files(input_dir):
    """List all model generation files in the specified directory that match a predefined regex pattern."""

    regex_pattern = re.compile(r'^generations_([\w\-\.\_]+)\.jsonl$')
    files = [(regex_pattern.match(filename).group(1), filename)
             for filename in os.listdir(input_dir) if regex_pattern.match(filename)]
    logging.info(f"Model generation files listed: {files}")
    return files



def load_generations_from_file(filepath):
    """Load generation data from a specified file, parsing each line as JSON."""

    with open(filepath, 'r') as f:
        generations = [json.loads(line) for line in f]
    logging.info(f"Generations loaded from {filepath}")
    return generations


def validate_answer(generation):
    """Validate whether the generated text predominantly indicates a 'yes' or 'no' answer, ignoring mixed or ambiguous responses."""

    # Using regex to find 'yes' or 'no' in any case within the generation
    yes_matches = re.findall(r"\byes\b", generation, re.IGNORECASE)
    no_matches = re.findall(r"\bno\b", generation, re.IGNORECASE)

    if yes_matches and not no_matches:
        return "Yes"
    elif no_matches and not yes_matches:
        return "No"
    else:
        logging.warning(f"NO clear YES or NO answer! {generation}")
        return "None"


def construct_content(q_id, datum, generation, question):
    """Construct the content to be sent to the OpenAI API based on input and generation details."""

    if q_id == 0:
        prefix = f"{SYSTEM_MESSAGE}\n\n" if not datum['input'].strip() else f"{SYSTEM_MESSAGE}\n\nInput:\n\"{datum['input']}\"\n\n"
        content = f"{prefix}Generated Text:\n\"{generation['output']}\"\n\nQuestion:\n{question}"
    else:
        content = f"Question:\n{question}"
    return content


def generate_answer(client, messages, model, temperature=0., seed=1234, max_retries=5):
    """Attempt to generate an answer using the OpenAI API, with retries on failure."""

    for _ in range(max_retries):
        try:
            completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature, seed=seed)
            answer = validate_answer(completion.choices[0].message.content)
            if answer != "None":
                return answer
        except Exception as e:
            logging.error(f"Error during completion generation: {e}", exc_info=True)
            time.sleep(10)
    return "None"


def evaluate(model, dataset, generations, eval_model='gpt-4-0613', temperature=0., seed=1234):
    """Evaluate the dataset with generated texts using the OpenAI API to answer decomposed questions."""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    evaluations = []
    for data, generation in tqdm(zip(dataset, generations), desc=f"{model}"):
        messages = []
        answers  = []
        decomposed_questions = [q.strip() for q in data['decomposed_questions'] if q.strip()]
        
        for q_id, question in enumerate(decomposed_questions):
            content = construct_content(q_id, data, generation, question)
            messages.append({"role": "user", "content": f"{content}\n"})
            
            answer = generate_answer(client, messages, model=eval_model, temperature=temperature, seed=seed)
            messages.append({"role": "assistant", "content": f"{answer}\n"})
            answers.append(answer)
        
        evals = {
            'input': data['input'],
            'output': generation['output'],
            'messages': messages,
            'answers': answers,
        }

        evaluations.append(evals)
        logging.info(f"Evaluations for data input: {data['input']} completed.")
    
    return evaluations


def main(args):
    dataset = load_dataset('kifai/KoInFoBench')['train']

    models_and_filenames = list_model_generation_files(args.input_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    result_filepath = os.path.join(args.output_dir, 'result.json')
    if os.path.exists(result_filepath):
        with open(result_filepath, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for model, filename in models_and_filenames:
        if model in results:
            continue
        filepath = os.path.join(args.input_dir, filename)
        generations = load_generations_from_file(filepath)
        evaluations = evaluate(model, dataset, generations, args.eval_model, args.temperature, args.seed)
        
        flattened_answers = [answer for evals in evaluations for answer in evals['answers']]
        
        # DRFR(Decomposed Requirements Following Ratio)
        drfr = sum(map(lambda x: 1 if x == "Yes" else 0, flattened_answers)) / len(flattened_answers)
        results[model] = {
            "filepath": filepath,
            "drfr": drfr,
            "evaluations": evaluations,
        }

    logging.info("Final evaluation results:")
    print(f"{'Model':<25} {'DRFR':>10}")
    print(f"{'-----':<25} {'----':>10}")
    for model, result in results.items():
        print(f"{model:<25} {result['drfr']:10.3f}")
        logging.info(f"Model: {model}, DRFR: {result['drfr']}")

    with open(result_filepath, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model outputs.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory where the model generations are stored.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to store the evaluation results.')
    parser.add_argument('--eval_model', type=str, default='gpt-4-0613', help='Model to use for evaluation.')
    parser.add_argument('--temperature', type=float, default=0., help='Temperature for model creativity. Lower value is more deterministic.')
    parser.add_argument('--seed', type=int, default=1234, help='If specified, such that repeated requests with the same seed and parameters should return the same result.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.output_dir, 'evaluations.log'), filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
