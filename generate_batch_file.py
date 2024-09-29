import json

'''
code for prepare two batch files, 'candidate/firm_batch_requests.jsonl'
'''
import json

def load_candidates(file_path):
    """Load candidate configuration from a JSON file."""
    with open(file_path) as f:
        candidates = json.load(f)
    # Only use the first 4 candidates for the small-scale test
    return candidates[:4]

def load_firm_vectors(file_path):
    """Load firm vectors from a text file, removing any non-numeric characters."""
    with open(file_path) as f:
        vectors = []
        for line in f:
            # Remove any non-numeric characters like brackets
            cleaned_line = line.strip().replace('[', '').replace(']', '')
            # Convert the cleaned line into a list of floats
            vector = list(map(float, cleaned_line.split()))
            vectors.append(vector)
    # Only use the first 4 firm vectors for the small-scale test
    return vectors[:4]

def generate_firm_descriptions(vectors, firm_types):
    """Generate descriptions for firms based on their vectors."""
    descriptions = []
    for i, vector in enumerate(vectors):
        description = f"F{i+1}: "
        description += ", ".join([f"{firm_types[j]} ({vector[j]*100:.0f}% emphasis)" for j in range(len(firm_types))])
        descriptions.append(description)
    return descriptions

def generate_candidate_prompts(candidates, firm_descriptions):
    """Generate prompts for candidates to choose a firm to apply to."""
    prompts = []
    for i, candidate in enumerate(candidates):
        candidate_details = candidate[f"Candidate_{i}"]
        prompt = (f"You are a {candidate_details['Sex']}, {candidate_details['Age']} candidate of {candidate_details['Race']} race."
                  f"\nHere are 4 firms to choose from:\n" + "\n".join(firm_descriptions) +
                  "\nChoose the firm you would like to apply to.")
        prompts.append(prompt)
    return prompts

def create_candidate_batch_requests(candidate_prompts, model="gpt-4o"):
    """Create batch requests for candidates."""
    batch_requests = []

    # Create requests for candidates
    for i, prompt in enumerate(candidate_prompts):
        request = {
            "custom_id": f"candidate-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a job candidate in this simulation."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000
            }
        }
        batch_requests.append(request)

    return batch_requests

def create_firm_batch_requests(firm_descriptions, model="gpt-4o"):
    """Create batch requests for firms."""
    batch_requests = []

    # Create requests for firms
    for i, firm_prompt in enumerate(firm_descriptions):
        request = {
            "custom_id": f"firm-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a firm in this simulation."},
                    {"role": "user", "content": firm_prompt + "\nYour mission is to choose a candidate to hire from the candidate pool each round. After you hire a candidate, you will receive feedback on their performance."}
                ],
                "max_tokens": 1000
            }
        }
        batch_requests.append(request)

    return batch_requests

def save_batch_requests_to_file(batch_requests, file_path):
    """Save the batch requests to a JSON Lines file."""
    with open(file_path, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')

def generate_batch_files():
    # File paths
    candidate_file = 'candidate_features.json'
    firm_vector_file = 'firms_vector.txt'
    candidate_output_file = 'candidate_batch_requests-small.jsonl'
    firm_output_file = 'firm_batch_requests-small.jsonl'

    # Firm types
    firm_types = ["technology", "management", "service", "production"]

    # Load data
    candidates = load_candidates(candidate_file)
    firm_vectors = load_firm_vectors(firm_vector_file)

    # Generate descriptions and prompts
    firm_descriptions = generate_firm_descriptions(firm_vectors, firm_types)
    candidate_prompts = generate_candidate_prompts(candidates, firm_descriptions)

    # Create batch requests
    candidate_batch_requests = create_candidate_batch_requests(candidate_prompts)
    firm_batch_requests = create_firm_batch_requests(firm_descriptions)

    # Save to separate files
    save_batch_requests_to_file(candidate_batch_requests, candidate_output_file)
    save_batch_requests_to_file(firm_batch_requests, firm_output_file)

    print(f"Candidate batch file created successfully: {candidate_output_file}")
    print(f"Firm batch file created successfully: {firm_output_file}")


# Run the main function
if __name__ == "__main__":
    generate_batch_files()