# imports

import os
import json
from random import choices
from unittest import case

from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import base64
from PIL import Image
from IPython.display import Audio, display
import tempfile
import subprocess
from io import BytesIO
from pydub import AudioSegment
import time
import anthropic
import google.generativeai
# Initialization

#load_dotenv(override=True)
load_dotenv(dotenv_path="C:/Users/milo.MILOJR-LENOVA/projects/llm_engineering/.env")

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthopic API Key exists and begins {anthropic_api_key[:8]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")



####defaults
MODEL = "gpt-4o-mini"
LANGUAGE = "English"
openai = OpenAI()
claude = anthropic.Anthropic()
#gemini = google.GenerativeAI()

#####

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += ("You always start the initial greeting by saying 'Welcome to flight AI, "
                   "How can I help you today?'. ")
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."
book_flight_description = ("Book a flight seat if it's available. Ask the user"
                           " for destination city and their name.")
book_flight_description += ("If no city is available. kindly inform the user of the"
                            " error, and offer to try again")
book_flight_description += ("If no seat are available, kindly inform the user that "
                            "their flight is full.")
book_flight_description += "if you don't know, say so"
book_flight_description += ("Once complete display the name, seat number "
                            "and flight number")

system_message += book_flight_description
# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
AVAILABLE_MODELS = {
                   "Anthropic": ["claude-3-7-sonnet-latest"],
                   "Google": ["gemini-2.0-flash","gemini-2.5-flash-preview-04-17"],
                   "OpenAI": ["gpt-4o-mini","o3-mini","gpt-4.1-nano","gpt-4.1"],
                   "Deep Seek":["deepseek-chat"],
                   "Deep Seek Reasoning" : ["deepseek-reasoner"]
}

#                    "Google-gemini-2.5-flash-preview-04-17": "gemini-2.5-flash-preview-04-17",

AVAILABLE_MODELS2 = {
                   "Anthropic-claude-3-7-sonnet-latest": "claude-3-7-sonnet-latest",
                   "Google-gemini-2.0-flash": "gemini-2.0-flash",
                    "OpenAI-gpt-4o-mini": "gpt-4o-mini",
                    "OpenAI-o3-mini": "o3-mini",
                    "OpenAI-gpt-4.1-nano": "gpt-4.1-nano",
                   "OpenAI-gpt-4.1" : "gpt-4.1"
}
TRANSLATION_LANGUAGE = { "English" : "English:",
                          "German": "German",
                          "Spanish":"Spanish",
                          "Japanese": "Japanese"}


# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}
return_model = {
    "name": "get_model2",
    "description": "Return the model to be used, as requested by the user and connection",
    "parameters": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "the model",
            },
        },
        "required": ["model"],
        "additionalProperties": False
    }
}

#"description": "Book a flight seat if it's available. Ask the user for destination city and their name. If no seat are available, kindly inform the user that their flight is full."
book_function = {
    "name": "book_the_flight",
        "description": "Book a flight seat if it's available. Ask the user for destination city and their name. If no seat are available, kindly inform the user that their flight is full. Once complete display the name, seat number and flight number",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "destination_city": {"type": "string"},
            "seat": {"type": "string"}
        },
        "required": ["name", "destination_city"],
        "additionalProperties": False
    }
}


# And this is included in a list of tools:

tools = [
            {"type": "function", "function": price_function},
            {"type": "function", "function": book_function},
            {"type": "function", "function": return_model}
        ]
seats =  {"seat 1": False,
          "seat 2": False
        }

MODEL_CHOICES = {
    "Anthropic-claude-3-7-sonnet-latest" : {
        "message=" : None,
        "max_tokens=":200,
        "temperature=": 0.7,
        "system=" : None,
    },
    "gpt-4o-mini": {
    # response = openai.chat.completions.create(model=model, messages=messages, tools=tools)
        "messages" : None,
        "tools" :"tools",
        "model" : "gpt-4o-mini",
    },
    "o3-mini" : {
        "messages" : None,
        "tools" :"tools",
        "model" : "o3-mini",
    },
    "gpt-4.1-nano" : {
        "messages" : None,
        "tools" :"tools",
        "model" : "gpt-4.1-nano",
    }


}
destination_flights =  {
    "london": {
        "Seat 1": None,
        "Seat 2": None,
         "seat" : ["Empty", "Empty"],
        "flight_no": 289
    },
    "paris": {
        "Seat 1": None,
        "Seat 2": None,
        "flight_no" : 325
    },
    "berlin": {
        "Seat 1": None,
        "Seat 2": None,
        "flight_no": 368
    },
    "new york": {
        "Seat 1": None,
        "Seat 2": None,
        "flight_no": 596
    }
}

########################################################################################

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

def get_model2(the_model):
    print (f"In get_model =  {AVAILABLE_MODELS[the_model]}")
    model = AVAILABLE_MODELS.get(the_model,"gpt-4o-mini")
    return model

def get_model(the_model):
    print (f"In get_model =  {AVAILABLE_MODELS2[the_model]}")
    return (AVAILABLE_MODELS2.get(the_model,"gpt-4o-mini"))

#def get_model(the_model):
#   print (f"In get_model =  {MODEL_CHOICES[the_model]["model"])
 #   print (")
 #   return (AVAILABLE_MODELS2.get(the_model,"gpt-4o-mini"))


def get_language(the_language):
        print(f"In get_languate =  {TRANSLATION_LANGUAGE[the_language]}")
        return(TRANSLATION_LANGUAGE[the_language])



def chat2(message, history):
    print (f"message: {message}")
    print (f"history: {history}")
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    print(f" reason = {response.choices[0].finish_reason}")
    if response.choices[0].finish_reason == "tool_calls":
        print("Chat: Got a tool_calls")
        message = response.choices[0].message
        print(f"the message is == {message}")
        response = handle_tool_call(message)
        # response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    print("talking")
    talker(response.choices[0].message.content)
    return response.choices[0].message.content

def get_response(model,user_messages,tools):
    print (f"In get_response:  {model}")
    print (f"       user_messages: {user_messages}")
    print (f"          tools: {tools}")
    match model:
        case "gpt-4o-mini" | "o3-mini" | "gpt-4.1-nano" | "gpt-4.1":
            return (openai.chat.completions.create(
                    model=model,
                    messages=user_messages,
                    tools=tools)
            )
        case "claude-3-7-sonnet-latest":
            print ("claude")
            claude_messages = []
            for msg in user_messages:
                print (msg)
                if msg["role"] != "system":  # Claude handles system message separately
                    claude_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            message = claude.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.7,
                system=system_message,
                messages=claude_messages

            )
               # messages = [
               #     {"role": "user", "content": user_messages},
               #     ],
                #)

            print (f"claude message.stop_reason =  {message.stop_reason}")
            return message
        case "gemini-2.0-flash"|"gemini-2.5-flash-preview-04-17":
                gem2=google.generativeai.GenerativeModel(
                    model_name=model,
                    system_instruction=system_message
                )

                return OpenAI(
                    api_key=google_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                ).chat.completions.create(model=model, messages=user_messages, tools=tools)


        case _ :
            return ({"error":"The model does not exist"})


def chat(history,model=MODEL, language = LANGUAGE):
    print (f"history: {history}")
    print (f"model={model}")
    print (f"Language={language}")
    messages = ([{"role": "system", "content": system_message}] +
                history +
                [{"role": "system", "content": f"You will provide all responses in the {language}"}]
                )
    try:

        response = openai.chat.completions.create(model=model, messages=messages, tools=tools )
    except Exception as e:
            print (f"There is is error:{e}")

    #return {"error": "Flight is full", "destination": city.title()
    print(f" reason = {response.choices[0].finish_reason}")
    match model:
       case "claude-3-7-sonnet-latest":
            print ("In case statement")
            print(f"reason = {response.content[0].text}")
            print(f" REASONS = {response.content[0]}")
            print(f" REASONS = {response.stop_reason}")
            if response.stop_reason == "tool_use":
                response = handle_tool_call(messages)
                messages.append(messages)
                messages.append(response)
                response = get_response(model, messages, tools)


    if response.choices[0].finish_reason == "tool_calls":
        print("Chat: Got a tool_calls")
        message = response.choices[0].message
        print(f"the message is == {message}")
        response = handle_tool_call(message)
        print (f"BACK FROM HANDLE_TOOL_CALL model=={model}")
        print (f"response = {response}")
        # response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        print (f"messages is == {messages}")
        response = openai.chat.completions.create(model=model, messages=messages)
    reply = response.choices[0].message.content
    print (f" reply = {reply}")
    history += [{"role":"assistant", "content": reply}]
    print("talking")
    talker(reply)
    #return response.choices[0].message.content
    return history

def chat4(history,model=MODEL, language = LANGUAGE):
    print (f"history: {history}")
    print (f"model={model}")
    print (f"Language={language}")
    messages = ([{"role": "system", "content": system_message}] +
                history +
                [{"role": "system", "content": f"You will provide all responses in the {language}"}]
                )
    try:

        response = openai.chat.completions.create(model=model, messages=messages, tools=tools )
    except Exception as e:
            print (f"There is is error:{e}")

    #return {"error": "Flight is full", "destination": city.title()
    print(f" reason = {response.choices[0].finish_reason}")

    if response.choices[0].finish_reason == "tool_calls":
        print("Chat: Got a tool_calls")
        message = response.choices[0].message
        print(f"the message is == {message}")
        response = handle_tool_call(message)
        print (f"BACK FROM HANDLE_TOOL_CALL model=={model}")
        print (f"response = {response}")
        # response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        print (f"messages is == {messages}")
        response = openai.chat.completions.create(model=model, messages=messages)
    reply = response.choices[0].message.content
    print (f" reply = {reply}")
    history += [{"role":"assistant", "content": reply}]
    print("talking")
    talker(reply)
    #return response.choices[0].message.content
    return history

def chat3(history,model=MODEL, language = LANGUAGE):
    print (f"In Chat:")
    print (f"history: {history}")
    print (f"model={model}")
    print (f"Language={language}")
    messages = ([{"role": "system", "content": system_message}] +
                history +
                [{"role": "system", "content": f"You will provide all responses in the {language} language"}]
                )
    print (f"messages is == {messages}")
    try:
        print ("going to get_response")
        response=get_response(model,messages,tools)
        print ("back from get_response")

    except Exception as e:
            print (f"There is the error:{e}")

    match model:
        case "claude-3-7-sonnet-latest":
            print ("In case statement")
            print(f"reason = {response.content[0].text}")
            print(f" REASONS = {response.content[0]}")
            print(f" REASONS = {response.stop_reason}")
            if response.stop_reason == "tool_use":
                response = handle_tool_call(messages)
                messages.append(messages)
                messages.append(response)
                response = get_response(model, messages, tools)
            reply = response.content[0].text
            print (f"Reply is == {reply}")
            history += [{"role": "assistant", "content": reply}]
            print("talking")
            talker(reply)
            # return response.choices[0].message.content
            return history
        case "gemini-2.5-flash-preview-04-17":
            print (response.choices[0].message.content)

    if response.choices[0].finish_reason == "tool_calls":
        print(f" reason = {response.choices[0].finish_reason}")
        print("Chat: Got a tool_calls")
        message = response.choices[0].message
        print(f"the message is == {message}")
        response = handle_tool_call(message)
        print (f"BACK FROM HANDLE_TOOL_CALL model=={model}")
        print (f"response = {response}")
        # response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
    print (f"messages is == {messages}")

    response = get_response(model, messages,tools)
    reply = response.choices[0].message.content
    print (f" reply = {reply}")
    history += [{"role":"assistant", "content": reply}]
    print("talking")
    talker(reply)
    #return response.choices[0].message.content
    return history


def book_the_flight2(name,city):
    city = city.lower()
    print (f" Plane to {city}")
    print (destination_flights[city])
    for key,value_info in destination_flights[city].items():
        print (f"key == {key}")
        if destination_flights[city][key] == None:
            print (f"Found empty seat Booking --> {city}  {key} = {name}")
            destination_flights[city][key] = name
            return destination_flights
        #if value_info[key_city] == None:
    return None

def book_the_flight(name,city):
    city = city.lower()
    print (f" Plane to {city}")
    if city not in destination_flights:
        #return {"error": f"No flight available to {city}"}
        return None
    print (destination_flights[city])


    #Checks if destination exists, returns error if not
    flight_info = destination_flights[city]
    print(f"flight_info = {flight_info}")

    # Find first available seat
    for seat_key in ["Seat 1", "Seat 2"]:
        if flight_info[seat_key] is None:
            print(f"Found empty seat. Booking --> {city} {seat_key} = {name}")
            print (f"Flight no = {flight_info['flight_no']}")
            flight_info[seat_key] = name
            return {
                "name": name,
                "seat": seat_key,
                "flight_no": flight_info["flight_no"],
                "destination": city.title(),
                "status": "confirmed"
            }
    return {"error": "Flight is full", "destination": city.title()}


def handle_tool_call(message):
    print("====handle_tool_call===")
    print (f"message = {message}")
    print(f"message.tools_calls = {message.tool_calls}")
    # tool_call = message.tool_calls[0]
    tool_call = message.tool_calls[0]
    print(f"printing tool_Call ")
    print(tool_call)
    func_name = tool_call.function.name
    print(f"Got function name == {func_name}")
    arguments = json.loads(tool_call.function.arguments)
    print (f"arguments = {arguments}")
    match func_name:
        case "get_ticket_price":
            city = arguments["destination_city"]
            price = get_ticket_price(city)

            response = {
                "role": "tool",
                "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call.id
             }
        case "book_the_flight":
            print("booking the flight")
            passenger = arguments["name"]
            city = arguments["destination_city"]
            print (f"booking flight for {passenger} to  {city}")
            flight = book_the_flight(passenger, city)
            print ("Back from book_the_flight")
            response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"booking": flight})
            }
            print (f"Return {response}")
            print ("===LEAVING TOOL HANDLE===")
            return response
        case "return_model":
            print("Choosing a model")
            themodel = arguments["model"]
            print(f"model to be used to  {themodel}")
            model=get_model(themodel)
            response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"themodel": model})
            }
            print("===LEAVING TOOL HANDLE===")
            return response, model
            print("===LEAVING TOOL HANDLE===")
    return response

def play_audio(audio_segment):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio.wav")
    try:
        audio_segment.export(temp_path, format="wav")
        time.sleep(
            3)  # Student Dominic found that this was needed. You could also try commenting out to see if not needed on your PC
        subprocess.call([
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-hide_banner",
            temp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass



def talker(message):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Also, try replacing onyx with alloy
            input=message
        )

        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play_audio(audio)
    except Exception as e:
            print (f"Error in tts:{e}")

print (AVAILABLE_MODELS2["OpenAI-gpt-4o-mini"])
#gr.ChatInterface(fn=chat, type="messages").launch()
with gr.Blocks() as ui:
    history = "Welcome to Flight AI"
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages", label="Flight AI response" )
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:", placeholder="Type your message here...")
        model_menu = gr.Dropdown(
                choices=AVAILABLE_MODELS2.keys(),
                label="Choose specific model",
                #value=AVAILABLE_MODELS2["OpenAI-gpt-4o-mini"]  ##Default
                value="Anthropic-claude-3-7-sonnet-latest"
        )
        language_menu = gr.Dropdown (
            choices=TRANSLATION_LANGUAGE,
            label = "Choose your language",
            value = "English"
        )
    with gr.Row():
        clear = gr.Button("Clear")
    def do_entry(message, history, model, language):
        print ("do_entry")

        if not message.strip():
            return "", history
        history += [{"role":"user", "content":message}]
        return "", history

    def chat_wrapper2(history, selected_model,selected_language):
        return chat(history, get_model(selected_model, get_language(selected_language)) if selected_model else MODEL)

    def chat_wrapper(history, selected_model,selected_language):
        if selected_language == None:
            language = LANGUAGE
        else:
            language=get_language(selected_language)

        if selected_model == None:
                model = MODEL
        else:
            model = get_model(selected_model)
        print (f"chat wrapper {history}")
        return chat(history, model, language)


    entry.submit(do_entry,
                 inputs=[entry, chatbot, model_menu,language_menu ],
                 outputs=[entry, chatbot]
                 ).then (
                    chat_wrapper,
                    inputs=[chatbot,model_menu, language_menu],
                    outputs=[chatbot]
                )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)