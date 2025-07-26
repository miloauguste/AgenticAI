# imports

import os
import json
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
# Initialization

#load_dotenv(override=True)
load_dotenv(dotenv_path="C:/Users/milo.MILOJR-LENOVA/projects/llm_engineering/.env")

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4o-mini"
openai = OpenAI()


system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."
book_flight_description = "Book a flight seat if it's available. Ask the user for destination city and their name."
book_flight_description += "If no seat are available, kindly inform the user that their flight is full."
book_flight_description += "Once complete display the name, seat number and flight number"

system_message += book_flight_description
# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

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
            {"type": "function", "function": book_function}
        ]
seats =  {"seat 1": False,
          "seat 2": False
        }

destination_flights =  {
    "london": {
        "Seat 1": None,
        "Seat 2": None,
         "seat" : ["Empty", "Empty"],
        "flight no": 289
    },
    "paris": {
        "Seat 1": None,
        "Seat 2": None,
        "flight no" : 325
    },
    "berlin": {
        "Seat 1": None,
        "Seat 2": None,
        "flight no": 368
    },
    "new york": {
        "Seat 1": None,
        "Seat 2": None,
        "flight no": 596
    }
}


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

def chat(history):
    print (f"history: {history}")
    messages = [{"role": "system", "content": system_message}] + history
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
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content": reply}]
    print("talking")
    talker(reply)
    return response.choices[0].message.content


def book_the_flight2(passenger, seatnumber, flight_number, destination_city):
    print ("in book_the_flight")
    city = destination_city.lower()
    flights[city][0]["name"] = passenger
    flights[city][0]["seat_number"] = seatnumber
    flights[city][0]["flight_number"] = flight_number
    return flights

def book_the_flight(name,city):
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
    if func_name == "get_ticket_price":
        city = arguments["destination_city"]
        price = get_ticket_price(city)

        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call.id
        }
        # return response, city
       # return response

    elif func_name == "book_the_flight":
        print("booking the flight")
        passenger = arguments["name"]
        city = arguments["destination_city"]
        print (f"booking flight for {passenger} to  {city}")
        flight = book_the_flight(passenger, city)
        response = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({"booking": flight})

        }
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
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",  # Also, try replacing onyx with alloy
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play_audio(audio)


gr.ChatInterface(fn=chat, type="messages").launch()