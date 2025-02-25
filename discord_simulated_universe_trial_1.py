import os
import discord
import asyncio
import datetime
import torch
import csv
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from dotenv import load_dotenv
import pytz
import re
import firebase_admin
from firebase_admin import credentials, firestore ,storage
import os
cred = credentials.Certificate("voice-samples-f2816-firebase-adminsdk-fbsvc-3a7a6eefc9.json")
firebase_admin.initialize_app(cred, {
   "storageBucket": "voice-samples-f2816.firebasestorage.app"

})

db = firestore.client()
bucket = storage.bucket() 

def check_and_create_user(user_id):
    # check if it is a new user
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        user_ref.set({"created_at": firestore.SERVER_TIMESTAMP}) # create file for the new user
    #else:
        # the user already exist

def check_and_create_role(user_id, role_name):
    # check if the requested character exist
    role_ref = db.collection("users").document(user_id).collection("roles").document(role_name)
    doc = role_ref.get()
    if not role_ref.get().exists:
        today_str = datetime.datetime.today().strftime("%Y-%m-%d")
        role_ref.set({
            "created_at": firestore.SERVER_TIMESTAMP,
            "first_day": today_str,
        })
    #else:
        # the character exist

def querydatabase(user_id,role_name,day_today):

    check_and_create_user(user_id)
    check_and_create_role(user_id, role_name)

    role_ref = db.collection("users").document(user_id).collection("roles").document(role_name)
    doc = role_ref.get()

    if not doc.exists:
        # role doesn't exist
        return None

    data = doc.to_dict()
    first_day_str = data.get("first_day")

    first_day = datetime.datetime.strptime(first_day_str, "%Y-%m-%d")
    today_date = datetime.datetime.strptime(day_today, "%Y-%m-%d")

    day_diff = (today_date - first_day).days
    return day_diff

def create_date_folder_and_upload_txt(user_id, role_name, date_str, file_path):
    
    if not os.path.exists(file_path):

        return None

    history_ref = db.collection("users").document(user_id).collection("roles").document(role_name).collection("history").document(date_str)

    blob = bucket.blob(f"chat_logs/{user_id}/{role_name}/{date_str}.txt")
    blob.upload_from_filename(file_path)

    history_ref.set({
        "chat_log_url": blob.public_url
    }, merge=True)

    return blob.public_url

EVENT_PROMPTS_FILE = "event_prompts.csv"  # Update the file path if needed

local_tz = pytz.timezone("America/Chicago")  # Change to your timezone
# Load environment variables (ensure DISCORD_BOT_TOKEN is set in your .env file)
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if DISCORD_BOT_TOKEN is None:
    raise ValueError("DISCORD_BOT_TOKEN environment variable not set.")

#######################################
# 1. Model & Tokenizer Initialization #
#######################################

max_seq_length = 4096
dtype = None
load_in_4bit = True

# Load the model & tokenizer (update the model path as needed)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/exx/Desktop/fine-tune/2B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

# Apply LoRA modifications with PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=1228,
    use_rslora=False,
    loftq_config=None,
)

# Update the tokenizer with the chat template (e.g., "llama-3.1")
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Enable inference optimizations
FastLanguageModel.for_inference(model)

##############################
# 2. Custom Text Streamer    #
##############################
class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.generated_tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.generated_tokens.append(token)


#########################################
# 3. Generation Function for Responses  #
#########################################
def generate_response_sync(conversation_history):
    """
    Generate a response using the conversation history.
    Debug print statements are added to show input details and full generated output.
    """
    print(conversation_history)
    # Prepare the input using the chat template
    inputs = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Create a streamer instance for the initial generation
    initial_streamer = CustomTextStreamer(tokenizer, skip_prompt=False)
    _ = model.generate(
        input_ids=inputs,
        attention_mask=(inputs != tokenizer.pad_token_id).to("cuda"),
        streamer=initial_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1
    )

    chat_input = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,  # Required for generation
        return_tensors="pt"
    ).to("cuda")

        # Compute the attention mask for the current input
    chat_attention_mask = (chat_input != tokenizer.pad_token_id).to("cuda")
    # Create a custom streamer to capture output tokens.
    streamer = CustomTextStreamer(tokenizer, skip_prompt=True)
    
    # Generate tokens; adjust max_new_tokens as needed.
    print("[DEBUG] Starting generation with max_new_tokens=100...")
    output = model.generate(
        input_ids=chat_input,
        attention_mask=chat_attention_mask,
        streamer=streamer,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    full_generated = tokenizer.decode(output[0], skip_special_tokens=False)
    print("[DEBUG] Full generated output:", full_generated)
    
    # Try to extract the assistant's response using a marker.
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    if assistant_marker in full_generated:
        response = full_generated.split(assistant_marker)[-1].strip()
        print("[DEBUG] Extracted response after assistant marker.")
    else:
        response = full_generated
        print("[DEBUG] No assistant marker found; using full output as response.")
    
    if not response:
        response = "I'm sorry, I didn't generate any response."
        print("[DEBUG] Response was empty, using fallback message.")
    
    print("[DEBUG] Final response to return:", response)
    return response  


#############################################
# 4. Session Management & Conversation Logs #
#############################################
# In-memory storage for user sessions.
sessions = {}
SESSION_TIMEOUT = 10 * 60  # 10 minutes in seconds
import discord
import datetime
import asyncio

SESSION_TIMEOUT = 10 * 60  # 10 minutes in seconds
LOG_FILENAME = "all_conversations.txt"

# Set the correct timezone (modify based on your location)
LOCAL_TIMEZONE = pytz.timezone("America/Chicago")  # Change this to your timezone

def get_current_time():
    """Returns the current time in the specified timezone."""
    now = datetime.datetime.now(LOCAL_TIMEZONE)
    return now.strftime("%I %p").lstrip("0"), now.strftime("%A")  # Example: ("4 PM", "Monday")

def load_character_prompt(character):
    """Loads the prompt from the CSV file based on the character and current day."""
    try:
        with open(EVENT_PROMPTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            current_time, current_day_name = get_current_time()
            current_day_int = datetime.datetime.now(LOCAL_TIMEZONE).day  # Get numerical day

            for row in reader:
                if row["Character"] == character:
                    # Ensure day formatting consistency
                    prompt = row["Prompt"].format(
                        Prompt="",
                        day=current_day_int,  # Ensure it correctly uses a numerical day
                        morning=row.get("morning", "{morning}"),  # Keeps {morning} if missing
                        afternoon=row.get("afternoon", "{afternoon}"),  
                        Evening=row.get("Evening", "{Evening}")  
                    )
                    prompt = prompt.replace("it is now 10 pm", f"it is now {current_time}")

                    # Remove any unwanted user tags
                    prompt = re.sub(r"\{\{user:\}\} Hi", "", prompt)  # Remove "{{user:}} Hi"

                    return {"role": "system", "content": prompt}

        return None  # No matching character found
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return None

def get_available_characters():
    """Reads the event prompts file and extracts unique character names."""
    characters = set()
    try:
        with open(EVENT_PROMPTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                characters.add(row["Character"])  # Collect character names
    except Exception as e:
        print(f"Error loading character list: {e}")
    return list(characters)

import json
import datetime

def save_dpo_preference(user_id, character, bad_response, correct_response):
    """Stores the conversation history in a structured JSON format for DPO training."""
    
    # Build the log entry as a dictionary
    log_entry = {
        "user": user_id,
        "character": character,
        "timestamp": datetime.datetime.now().isoformat(),
        "full_context": [],
        "bad_response": bad_response,
        "correct_response": correct_response
    }
    
    # Retrieve conversation history up until the bad response
    if user_id in sessions:
        for msg in sessions[user_id]["conversation_history"]:
            log_entry["full_context"].append({
                "role": msg["role"],
                "content": msg["content"]
            })
            if msg["content"] == bad_response:
                # Add an explicit marker for the bad response
                log_entry["full_context"].append({
                    "marker": "🔴 [BAD RESPONSE] <-- Marked for correction"
                })
                break
    
    # Write the log entry as a single JSON object per line (JSONL format)
    with open("dpo_log.jsonl", "a", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False)
        f.write("\n")
    
    print(f"[DPO] Logged full conversation history for user {user_id} in JSON format.")



def save_session(user_id, session,target):
    """
    Append the conversation history to a single log file.
    """
    with open(LOG_FILENAME, "a", encoding="utf-8") as f:
        f.write(f"\n--- Session with user {user_id} started at {session['start_time']} ---\n")
        for msg in session["conversation_history"]:
            f.write(f"{msg['role']}: {msg['content']}\n")
        f.write(f"--- Session ended at {datetime.datetime.now()} ---\n")
    Today_date = datetime.datetime.today().strftime("%Y-%m-%d")
    value = querydatabase(user_id,target,Today_date)
    

async def session_timeout_checker():
    """
    Periodically check sessions and terminate those inactive for more than 10 minutes.
    """
    await client.wait_until_ready()
    while not client.is_closed():
        now = datetime.datetime.now()
        to_remove = []
        for user_id, session in sessions.items():



            if (now - session["last_activity"]).total_seconds() > SESSION_TIMEOUT:
                target = session["target"]  # Retrieve target before calling save_session()
                save_session(user_id, session, target)
                to_remove.append(user_id)
        for user_id in to_remove:
            del sessions[user_id]
        await asyncio.sleep(60)  # Check every minute

# Discord Bot Setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    client.loop.create_task(session_timeout_checker())

@client.event
async def on_message(message):
    if message.author == client.user:
        return


    content = message.content.strip()
    user_id = str(message.author.id)
    if content.lower() == "!characters":
        available_characters = get_available_characters()
        await message.channel.send(f"Characters with events: {', '.join(available_characters)}")
        return
    
    if content.lower() == "!bad":
        if user_id in sessions and "last_response" in sessions[user_id]:
            sessions[user_id]["dpo_pending"] = sessions[user_id]["last_response"]
            await message.channel.send("Understood. Your next message will be recorded as the preferred response.")
        else:
            await message.channel.send("There's no previous response to mark as bad.")
        return
    if user_id in sessions:
        # If the user previously marked a response as bad, use the new message as the correct response
        if "dpo_pending" in sessions[user_id]:
            bad_response = sessions[user_id].pop("dpo_pending")  # Remove the flag after use
            correct_response = content

            # 🔥 Save the full conversation context before correction
            save_dpo_preference(user_id, sessions[user_id]["target"], bad_response, correct_response)

            await message.channel.send("Your correction has been recorded for model training.")

            # Replace bad response with correct one in history
            for i in range(len(sessions[user_id]["conversation_history"])):
                if sessions[user_id]["conversation_history"][i]["content"] == bad_response:
                    sessions[user_id]["conversation_history"][i]["content"] = correct_response
                    break

            # Track the corrected response as the last assistant response
            sessions[user_id]["last_response"] = correct_response

            return





    if content.lower() in ["exit", "quit"]:
        if user_id in sessions:
            save_session(user_id, sessions[user_id],sessions[user_id]["target"])
            del sessions[user_id]
            await message.channel.send("Session ended and saved.")
        else:
            await message.channel.send("No active session to end.")
        return

    if user_id not in sessions:
        check_and_create_user(user_id)
        if content.startswith("!ask"):
            parts = content.split(" ", 2)  # Ensure full character name is captured
            if len(parts) < 2:
                await message.channel.send("Please specify a target character. Example: `!ask Yan_Qing Hello`")
                return
            
            target = parts[1]  # Capture full character name
            new_content = parts[2] if len(parts) > 2 else ""
            check_and_create_role(user_id, target)

            # Debug print to verify character detection
            print(f"[DEBUG] Detected character: {target}")

            # Load the appropriate prompt for the character
            system_prompt = load_character_prompt(target)
            if not system_prompt:
                await message.channel.send(f"No schedule found for {target}. Available characters: `!characters`")
                return

            sessions[user_id] = {
                "conversation_history": [system_prompt],
                "last_activity": datetime.datetime.now(),
                "start_time": datetime.datetime.now(),
                "target": target
            }

            if new_content:
                sessions[user_id]["conversation_history"].append({"role": "user", "content": new_content})

            await message.channel.send(f"Starting a new session with {target}...")
        else:
            return
    else:
        # Continue the active session
        sessions[user_id]["conversation_history"].append({"role": "user", "content": content})
        sessions[user_id]["last_activity"] = datetime.datetime.now()

    await message.channel.send("Thinking... 🤔")

    try:
        print("[DEBUG] About to generate response...")
        assistant_reply = await asyncio.to_thread(generate_response_sync, sessions[user_id]["conversation_history"])
        print("[DEBUG] Received assistant reply:", assistant_reply)

        sessions[user_id]["conversation_history"].append({"role": "assistant", "content": assistant_reply})
        
        sessions[user_id]["last_response"] = assistant_reply  
        
        await message.channel.send(f"**{sessions[user_id]['target']}:** {assistant_reply}")
    except Exception as e:
        await message.channel.send(f"An error occurred: {str(e)}")

client.run(DISCORD_BOT_TOKEN)
