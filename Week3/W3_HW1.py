
import os

import requests
from openai import OpenAI
import gradio as gr
from IPython.display import Markdown, display, update_display
from huggingface_hub import login
#from google.colab import userdata

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

import torch
import json
import re
import pandas as pd
import openai
import io
import gc
import anthropic
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
from dotenv import load_dotenv
#from Week3.simplified_synthetic_data import show_available_models

load_dotenv(dotenv_path="C:/Users/milo.MILOJR-LENOVA/projects/llm_engineering/.env")
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
#### INITIALIZATION ###
#anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
loaded_hf_models = {}  # Cache for loaded HF models

anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
openai_client = openai.OpenAI(api_key=openai_api_key)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

def generate(model, messages):
  tokenizer = AutoTokenizer.from_pretrained(model)
  tokenizer.pad_token = tokenizer.eos_token
  inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
  streamer = TextStreamer(tokenizer)
  model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quant_config)
  outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)
  del model, inputs, tokenizer, outputs, streamer
  gc.collect()
  torch.cuda.empty_cache()

def get_system_prompt():
    system_prompt = """You are a synthetic data generator. Your job is to create realistic, diverse test data based on user specifications.

    Rules:
    - Generate data that looks authentic and realistic
    - Make each sample unique and different from others
    - Follow the exact format requested by the user
    - Create data suitable for testing purposes
    - Be creative but stay relevant to the requested domain"""
    return system_prompt

def create_dataset_prompt(dataset_type, user_description, num_samples):
    system_prompt = get_system_prompt()
    user_prompt = f"""
    The business problem is : {user_description}. \n
    Generate {num_samples} samples of {dataset_type}.
    Description: {user_description}
    The Format: must be: csv

    Generate only the data, no explanations."""

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    return full_prompt

#######################################################################################

HUGGINGFACE_MODELS2 = {
   "FLAN-T5 Base (Recommended)": "google/flan-t5-base",
   "GPT-2 Medium": "gpt2-medium",
   "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.1"
}
HUGGINGFACE_MODELS = {
   "PHI3": "microsoft/Phi-3-mini-4k-instruct",
   "Gemma2" : "google/gemma-2-2b-it",
   "Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}
FRONTIER_MODELS = {
   "GPT-4 (OpenAI)": "openai/gpt-4",
   "Claude-3 (Anthropic)": "anthropic/claude-3",
   "Gemini Pro (Google)": "google/gemini-pro"
}


ALL_MODELS = {**HUGGINGFACE_MODELS, **FRONTIER_MODELS}

DATASET_TYPES = {
   "Product Descriptions": "Generate product descriptions for e-commerce",
   "Job Postings": "Generate job posting descriptions for hiring",
   "Customer Reviews": "Generate customer reviews for products/services",
   "User Profiles": "Generate user profile information",
   "Business Emails": "Generate professional business email content",
   "Promissorry notes" : "Generate professional promissorry notes",
   "Custom": "Custom dataset type (specify in description)"
}

##################################################################################

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
HF_MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
FRONTIER_MODEL = "openai/gpt-4"
current_model = "openai/gpt-4"
current_tokenizer = None
current_pipeline = None
####################################################################################
def load_huggingface_model(model_name):
   """Load a HuggingFace model for text generation"""
   try:
       global current_model, current_tokenizer, current_pipeline

       print(f"🔄 Loading model: {model_name}")
       print(f"🔧 CUDA available: {torch.cuda.is_available()}")
       
       # Login to HuggingFace
       login(HF_TOKEN, add_to_git_credential=True)
       print("✅ HuggingFace login successful")

       # Common pipeline configuration
       device = 0 if torch.cuda.is_available() else -1
       common_config = {
           "device": device,
           "max_length": 512,
           "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
           "trust_remote_code": True
       }
       
       print(f"🎯 Device: {'GPU' if device == 0 else 'CPU'}")
       print(f"🎯 Torch dtype: {common_config['torch_dtype']}")

       if "Phi-3-mini-4k-instruct" in model_name:
           print("🤖 Loading Phi-3 model...")
           current_pipeline = pipeline("text-generation", model=model_name, **common_config)

       elif "gemma-2-2b-it" in model_name:
           print("🤖 Loading Gemma-2 model...")
           current_pipeline = pipeline("text-generation", model=model_name, **common_config)
           
       elif "Meta-Llama-3.1-8B-Instruct" in model_name:
           print("🤖 Loading Llama-3.1 model...")
           current_pipeline = pipeline("text-generation", model=model_name, **common_config)
       else:
           return f"❌ Unsupported model: {model_name}"

       print("✅ Model loaded successfully!")
       return current_pipeline

   except Exception as e:
       error_msg = f"❌ Error loading HF model {model_name}: {str(e)}"
       print(error_msg)
       return error_msg


#############################################################################
def huggingface_call(model_name,prompt, dt="cvs", num_samples = 10 ):
    pipe = load_huggingface_model(model_name)
    if isinstance(pipe, str):
        return pipe  # Return error string if model loading failed
    full_prompt = create_dataset_prompt(dt, prompt, num_samples)
    results = pipe(full_prompt, max_new_tokens=200, pad_token_id=pipe.tokenizer.eos_token_id)
    return results

def model_call(prompt,model_name):
    print ("Inference to FT model")
    model_name=model_name.split('/')[-1]
    print (f"model_name = {model_name}")
    print (f"prompt = {prompt}")
    """Simulate frontier model API calls (implement actual API calls here)"""
    # This is a placeholder - you would implement actual API calls to OpenAI, Anthropic, etc.
    sample_responses = [
           "High-quality wireless headphones with noise cancellation technology",
           "Premium smartphone with advanced camera system and long battery life",
           "Professional laptop designed for creative professionals and developers",
           "Ergonomic office chair with lumbar support and adjustable features",
           "Smart home security system with 24/7 monitoring capabilities"
       ]
    output_tokens = 100
    try:
        if model_name == "Meta-Llama-3.1-8B-Instruct" or model_name == "gemma-2-2b-it":
            print (f"Loading huggingface model: {model_name}")
            results = huggingface_call(model_name,prompt,"csv",10,)
            #pipe = load_huggingface_model(model_name)
            #full_prompt = create_dataset_prompt("cvs",prompt, 10)
            #result = pipe(full_prompt, max_new_tokens=200, pad_token_id=pipe.tokenizer.eos_token_id)
            return results[0]['generated_text'][len(prompt):].strip()

    except Exception as e:
        return f"Error: with HF model: {str(e)}"


    if model_name == 'claude-3':
        try:
            prompt = create_dataset_prompt("csv", prompt, 10)
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=output_tokens,
                system=get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        except Exception as e:
            return f"Error generating dataset: {str(e)}"

        return response.content[0].text

    elif model_name == 'gpt-4':
        print ("openai gpt-4")
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": create_dataset_prompt("csv", prompt, 10)}
        ]

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages = messages,
                max_tokens=output_tokens,
            )
        except Exception as e:
            return f"Error generating data {str(e)}"

        return response.choices[0].message.content

######################### GEN DATA SET #####################################
def gen_dataset(problem, hf, model):
    print (f"gen_dataset problem = {problem}, hf = {hf}, model = {model})")
    return model_call(problem, model)


def get_models(hf_key, model_key):
    """
    Safe model retrieval with fallbacks and validation
    """
    try:
        # Get HF model with fallback
        hf_model = HUGGINGFACE_MODELS.get(hf_key, HF_MODEL)

        # Get frontier model with fallback
        frontier_model = FRONTIER_MODELS.get(model_key, FRONTIER_MODEL)

        # Log selections for debugging
        print(f"Selected HF: {hf_key} -> {hf_model}")
        print(f"Selected Frontier: {model_key} -> {frontier_model}")

        return hf_model, frontier_model

    except Exception as e:
        print(f"Error in get_models: {e}")
        return HF_MODEL, FRONTIER_MODEL

##########################UI###################################################
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, label="Data Set generator")
    with gr.Row():
        biz_problem = gr.Textbox(label="Chat with our Data Set  Assistant:", placeholder="Type your message here...")
        model_menu = gr.Dropdown(
            choices=FRONTIER_MODELS.keys(),
            label="Choose specific model",
            # value=AVAILABLE_MODELS2["OpenAI-gpt-4o-mini"]  ##Default
            # value="Anthropic-claude-3-7-sonnet-latest"
        )
        hf_menu = gr.Dropdown(
            choices=HUGGINGFACE_MODELS,
            label="Choose your HF model",
            value="Gemma2"
        )
    with gr.Row():
        model_type = gr.Radio(
            choices=["Frontier Models", "HuggingFace Models"],
            value="Frontier Models",
            label="Model Type"
        )

        # Update the event binding to include model_type


    with gr.Row():
        clear = gr.Button("Clear")


    def chat_response_with_type(message, history, frontier_model, hf_model, model_type):
        if not message.strip():
            return history, ""

            # Choose which model to use
        if model_type == "HuggingFace Models":
            response = model_call(message, HUGGINGFACE_MODELS[hf_model])
        else:
            response = gen_data_wrapper(message, frontier_model, hf_model)
        history.append([message, response])
        return history, ""


    def do_entry(biz_problem, history, model):

        print("do_entry")

        if not biz_problem.strip():
            return "", history
        history += [{"role": "user", "content": biz_problem}]
        return "", history


    def gen_data_wrapper(problem, selected_model, selected_hf):
        hf_model, model = get_models(selected_hf,
                                     selected_model)

        # These null checks are now redundant but kept for safety
        if hf_model is None:
            hf_model = HF_MODEL
        if model is None:
            model = FRONTIER_MODEL

        print(f"Using frontier model: {model}")
        print(f"Using HF model: {hf_model}")
        print(f"Problem: {problem}")
        return gen_dataset(problem, hf_model, model)


    def chat_response(message, history, model, hf_model):
        """Process user message and return updated history"""
        if not message.strip():
            return history,""

        # Generate dataset
        response = gen_data_wrapper(message, model, hf_model)

        # Append to history as [user_message, bot_response] pairs
        history.append([message, response])

        return history,""

        # Event binding


    biz_problem.submit(
        chat_response_with_type,
        inputs=[biz_problem, chatbot, model_menu, hf_menu, model_type],
        outputs=[chatbot, biz_problem]
    )

    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(debug=True, share = False, inbrowser=True)