# Import necessary libraries from the transformers package
# - AutoModelForCausalLM: for loading the causal language model
# - AutoTokenizer: for tokenizing input text
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import torch for PyTorch operations
import torch

# Import time module to measure execution time
import time

# Import os module to interact with the operating system (e.g., environment variables)
import os

# Set the environment variable 'USE_LADE' to '1' to enable Lookahead Decoding (LADE)
# This must be done before importing and configuring the 'lade' module
os.environ["USE_LADE"] = "1"

# Import the LADE (Lookahead Decoding) library
import lade

# Apply LADE augmentations to all necessary functions in the Transformers library
# This function modifies certain methods to integrate LADE into the generation process
lade.augment_all()

# Configure LADE parameters for optimal performance
# - LEVEL: Controls the level of parallelism (higher levels may increase speed but use more memory)
# - WINDOW_SIZE: The maximum number of tokens considered in the lookahead window
# - GUESS_SET_SIZE: The size of the guess set for potential next tokens
# - DEBUG: Set to 1 to enable debug output (0 to disable)
# - POOL_FROM_PROMPT: If True, pools embeddings from the prompt for efficiency
lade.config_lade(
    LEVEL=7,
    WINDOW_SIZE=20,
    GUESS_SET_SIZE=20,
    DEBUG=1,
    POOL_FROM_PROMPT=True
)

# Ensure that a CUDA-enabled GPU is available
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU settings."

# Set the device for PyTorch computations to GPU
torch_device = "cuda"

# Specify the name or path of the pre-trained language model
# Note: Ensure that the model is compatible with LADE
# LADE currently supports LLaMA models (e.g., 'meta-llama/Llama-2-7b-hf')
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained language model with half-precision (fp16) for faster inference
# The 'device_map' argument ensures the model is loaded onto the specified device (GPU in this case)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half-precision floating point for faster computation
    device_map=torch_device     # Load the model onto the GPU
)

# Assign the tokenizer to the model (some models may require this explicit assignment)
model.tokenizer = tokenizer

# Define the prompt or input text for the model
prompt = "What can you help me with? Tell me in detail"

# Construct the full input text, including any special tokens or formatting required by the model
input_text = (
    f"<|system|>\nYou are a friendly chatbot. </s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
)

# Tokenize the input text and convert it into tensors suitable for the model
# Move the input tensors to the GPU device
model_inputs = tokenizer(input_text, return_tensors='pt').to(torch_device)

# Warm up the model by generating a single token
# This helps to reduce the initial overhead and stabilize generation timings
greedy_output = model.generate(**model_inputs, max_new_tokens=1)

# Synchronize CUDA operations to ensure accurate timing measurements
# This ensures that all previous CUDA commands have completed
torch.cuda.synchronize()

# Measure the time taken for sampled generation (with LADE enabled)
# Start the timer
t0s = time.time()

# Generate 256 new tokens using sampling strategies
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=256,   # Generate up to 256 new tokens
    do_sample=True,       # Enable sampling (as opposed to deterministic greedy decoding)
    temperature=0.7,      # Temperature parameter for controlling randomness (lower is less random)
    top_k=50,             # Top-K sampling (consider the top 50 tokens at each step)
    top_p=0.9             # Nucleus (Top-P) sampling (consider tokens with cumulative probability up to 0.9)
)

# Synchronize CUDA operations to ensure all computations are complete
torch.cuda.synchronize()

# Stop the timer for sampling
t1s = time.time()

# Measure the time taken for greedy (deterministic) generation
# Start the timer
t0g = time.time()

# Generate 256 new tokens using greedy decoding (without sampling)
greedy_output = model.generate(
    **model_inputs,
    max_new_tokens=256,  # Generate up to 256 new tokens
    do_sample=False      # Disable sampling for deterministic output
)

# Synchronize CUDA operations
torch.cuda.synchronize()

# Stop the timer for greedy decoding
t1g = time.time()

# Print a separator for clarity in the output
print("Output:\n" + 100 * '-')

# Decode and print the output from greedy decoding
# 'skip_special_tokens=False' retains any special tokens in the output
print("Greedy output: ", tokenizer.decode(greedy_output[0], skip_special_tokens=False))

# Decode and print the output from sampling
print("Sample output: ", tokenizer.decode(sample_output[0], skip_special_tokens=False))

# Calculate the number of tokens generated during greedy decoding
# Subtract the number of input tokens from the total output tokens
greedy_gen_tokens = greedy_output.numel() - model_inputs['input_ids'].numel()

# Calculate the generation speed (tokens per second) for greedy decoding
greedy_gen_speed = greedy_gen_tokens / (t1g - t0g)

# Calculate the number of tokens generated during sampling
sample_gen_tokens = sample_output.numel() - model_inputs['input_ids'].numel()

# Calculate the generation speed (tokens per second) for sampling
sample_gen_speed = sample_gen_tokens / (t1s - t0s)

# Print generation statistics for greedy decoding
print(
    f"Greedy Generated Tokens: {greedy_gen_tokens} Generation Speed: {greedy_gen_speed:.2f} tokens/s"
)

# Print generation statistics for sampling
print(
    f"Sample Generated Tokens: {sample_gen_tokens} Generation Speed: {sample_gen_speed:.2f} tokens/s"
)
