import json
import os
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import random
import time
import logging
from dataclasses import dataclass, asdict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configure logging (file only, no terminal output)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('npc_chat.log')
    ]
)

@dataclass
class PlayerMessage:
    player_id: int
    text: str
    timestamp: str
    
    def get_datetime(self) -> datetime:
        return datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))

@dataclass
class PlayerState:
    player_id: int
    mood: str = "neutral"
    previous_messages: deque = None
    interaction_count: int = 0
    current_topic: str = "general"
    mood_history: List[str] = None
    
    def __post_init__(self):
        if self.previous_messages is None:
            self.previous_messages = deque(maxlen=3)
        if self.mood_history is None:
            self.mood_history = ["neutral"]

@dataclass
class NPCResponse:
    player_id: int
    player_message: str
    npc_reply: str
    mood: str
    previous_messages: List[str]
    interaction_count: int
    topic: str
    timestamp: str
    fallback_used: bool = False

class NPCChatSystem:
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_client = None
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                logging.info("Groq client initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize Groq client: {e}")
        
        self.player_states: Dict[int, PlayerState] = {}
        self.responses: List[NPCResponse] = []
        
        # Fallback responses organized by mood and topic
        self.fallback_responses = {
            "neutral": {
                "greeting": ["Hello there, traveler!", "Greetings!", "Welcome to our village."],
                "quest": ["I might have something for you to do.", "Check the quest board.", "Perhaps later."],
                "help": ["I'll do what I can to assist.", "What do you need?", "How may I help you?"],
                "general": ["Interesting...", "I see.", "Tell me more.", "That's worth considering."],
                "farewell": ["Safe travels!", "Until next time.", "Farewell, friend."]
            },
            "friendly": {
                "greeting": ["Wonderful to see you again!", "My friend! How are you?", "Always a pleasure!"],
                "quest": ["I have just the thing for someone like you!", "You're exactly who I was hoping to see!", "I've been saving this quest for you."],
                "help": ["Of course! I'm happy to help.", "Anything for a friend like you!", "I'd be delighted to assist."],
                "general": ["That sounds fascinating!", "You're quite wise!", "I couldn't agree more!"],
                "farewell": ["Come back soon, my friend!", "I'll miss our conversations!", "Until we meet again!"]
            },
            "angry": {
                "greeting": ["What do you want now?", "*glares*", "You again..."],
                "quest": ["I don't have anything for you.", "Find someone else.", "You're not ready for my quests."],
                "help": ["Figure it out yourself.", "I'm busy.", "Why should I help you?"],
                "general": ["Whatever.", "*scoffs*", "I don't care.", "That's ridiculous."],
                "farewell": ["Good riddance.", "Don't come back soon.", "*turns away*"]
            }
        }
        
        # Topic keywords for classification
        self.topic_keywords = {
            "greeting": ["hello", "hi", "greetings", "hey", "good morning", "good day"],
            "quest": ["quest", "mission", "task", "job", "adventure", "do you have"],
            "help": ["help", "assist", "support", "guide", "show me", "how to", "where"],
            "farewell": ["bye", "goodbye", "farewell", "see you", "later", "leaving"],
            "insult": ["stupid", "useless", "idiot", "dumb", "worthless", "hate", "suck"],
            "compliment": ["thank", "great", "awesome", "wonderful", "amazing", "good job"]
        }

    def classify_topic(self, message: str) -> str:
        """Classify message topic based on keywords"""
        message_lower = message.lower()
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return topic
        return "general"

    def determine_mood_change(self, message: str, current_mood: str, topic: str) -> str:
        """Determine mood change based on message content and topic"""
        message_lower = message.lower()
        
        # Strong mood triggers
        if topic == "insult" or any(word in message_lower for word in ["stupid", "useless", "hate", "idiot"]):
            return "angry"
        elif topic == "compliment" or any(word in message_lower for word in ["thank", "great", "awesome"]):
            return "friendly"
        
        # Gradual mood shifts
        if current_mood == "neutral":
            if any(word in message_lower for word in ["please", "help", "quest", "friend"]):
                return "friendly" if random.random() > 0.7 else "neutral"
            elif any(word in message_lower for word in ["annoying", "boring", "whatever"]):
                return "angry" if random.random() > 0.8 else "neutral"
        elif current_mood == "friendly":
            if topic == "insult":
                return "neutral"  # Friendly NPCs are more forgiving
        elif current_mood == "angry":
            if topic == "compliment" and random.random() > 0.6:
                return "neutral"  # Chance to calm down
        
        return current_mood

    def generate_npc_response(self, player_message: str, player_state: PlayerState, topic: str) -> Tuple[str, bool]:
        """Generate NPC response using Groq API with fallback"""
        fallback_used = False
        
        # Prepare context from previous messages
        context = ""
        if player_state.previous_messages:
            context = "Previous conversation:\n" + "\n".join([
                f"Player: {msg}" for msg in list(player_state.previous_messages)
            ]) + "\n\n"
        
        # Create prompt for Groq
        prompt = f"""You are a fantasy NPC (non-player character) in a medieval village. 

Character context:
- Current mood: {player_state.mood}
- Topic: {topic}
- Interaction count: {player_state.interaction_count}
- Player ID: {player_state.player_id}

{context}Current player message: "{player_message}"

Instructions:
- Keep responses SHORT (1-2 sentences max)
- Match the mood: neutral (professional), friendly (warm/helpful), angry (curt/dismissive)
- Stay in character as a medieval fantasy NPC
- Reference previous context if relevant
- Don't break character or mention being an AI

Generate only the NPC's response:"""

        # Try Groq API first
        if self.groq_client:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful fantasy NPC who gives short, contextual responses."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=30
                )
                response = chat_completion.choices[0].message.content.strip()
                logging.info(f"Generated response via Groq for player {player_state.player_id}")
                return response, fallback_used
                
            except Exception as e:
                logging.warning(f"Groq API failed: {e}, using fallback")
                fallback_used = True
        else:
            fallback_used = True
            
        # Fallback to predefined responses
        mood_responses = self.fallback_responses.get(player_state.mood, self.fallback_responses["neutral"])
        topic_responses = mood_responses.get(topic, mood_responses["general"])
        response = random.choice(topic_responses)
        
        logging.info(f"Used fallback response for player {player_state.player_id}")
        return response, fallback_used

    def process_message(self, message: PlayerMessage) -> NPCResponse:
        """Process a single player message and generate NPC response"""
        player_id = message.player_id
        
        # Get or create player state
        if player_id not in self.player_states:
            self.player_states[player_id] = PlayerState(player_id=player_id)
        player_state = self.player_states[player_id]
        
        # Classify message topic
        topic = self.classify_topic(message.text)
        
        # Update mood based on message
        new_mood = self.determine_mood_change(message.text, player_state.mood, topic)
        if new_mood != player_state.mood:
            player_state.mood_history.append(new_mood)
            player_state.mood = new_mood
            logging.info(f"Player {player_id} mood changed to {new_mood}")
        
        # Update interaction count and topic
        player_state.interaction_count += 1
        player_state.current_topic = topic
        
        # Generate NPC response
        try:
            npc_reply, fallback_used = self.generate_npc_response(message.text, player_state, topic)
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            npc_reply, fallback_used = random.choice(self.fallback_responses[player_state.mood]["general"]), True
        
        # Create response object
        response = NPCResponse(
            player_id=player_id,
            player_message=message.text,
            npc_reply=npc_reply,
            mood=player_state.mood,
            previous_messages=list(player_state.previous_messages),
            interaction_count=player_state.interaction_count,
            topic=topic,
            timestamp=message.timestamp,
            fallback_used=fallback_used
        )
        
        # Update player state with current message
        player_state.previous_messages.append(message.text)
        
        return response

    def simulate_groq_response(self, player_message: str, player_state: PlayerState, topic: str) -> Tuple[str, bool]:
        """Simulate Groq API response (replace with actual API call in real implementation)"""
        # This simulates the API call - in real implementation, call the actual Groq API
        if random.random() > 0.1:  # 90% success rate simulation
            # Simulate AI-generated response based on mood and context
            if player_state.mood == "friendly":
                responses = [
                    "I'm glad you asked! Let me help you with that.",
                    "Of course, my friend! I'd be happy to assist.",
                    "That's a great question! Here's what I know..."
                ]
            elif player_state.mood == "angry":
                responses = [
                    "What do you expect me to do about it?",
                    "*sighs heavily* Fine, I suppose I can tell you.",
                    "This better be important..."
                ]
            else:  # neutral
                responses = [
                    "I see. Let me think about that.",
                    "That's an interesting point.",
                    "I can help you with that matter."
                ]
            
            return random.choice(responses), False
        else:
            # Simulate API failure
            raise Exception("Simulated API failure")

    def load_messages(self, filename: str) -> List[PlayerMessage]:
        """Load and parse messages from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            messages = [PlayerMessage(**msg) for msg in data]
            # Sort by timestamp to process chronologically
            messages.sort(key=lambda x: x.get_datetime())
            
            logging.info(f"Loaded {len(messages)} messages from {filename}")
            return messages
            
        except FileNotFoundError:
            logging.error(f"File {filename} not found")
            return []
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return []

    def process_all_messages(self, messages: List[PlayerMessage]):
        """Process all messages in chronological order"""
        logging.info(f"Processing {len(messages)} messages...")
        
        for i, message in enumerate(messages, 1):
            try:
                response = self.process_message(message)
                self.responses.append(response)
                
                # Log the interaction
                self.log_interaction(response)
                
                # Small delay to simulate real processing
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error processing message {i}: {e}")
        
        logging.info(f"Completed processing all messages. Generated {len(self.responses)} responses.")

    def log_interaction(self, response: NPCResponse):
        """Log individual interaction"""
        logging.info(f"""
        Player {response.player_id} | Interaction #{response.interaction_count}
        Message: "{response.player_message}"
        NPC Reply: "{response.npc_reply}"
        Mood: {response.mood} | Topic: {response.topic}
        Previous Messages: {response.previous_messages}
        Fallback Used: {response.fallback_used}
        Timestamp: {response.timestamp}
        {'='*50}
        """)

    def save_results(self, filename: str = "npc_chat_results.json"):
        """Save all results to JSON file"""
        results = []
        for response in self.responses:
            results.append(asdict(response))
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {filename}")

    def generate_summary_report(self):
        """Generate summary statistics"""
        if not self.responses:
            return
        
        total_interactions = len(self.responses)
        players = set(r.player_id for r in self.responses)
        fallback_count = sum(1 for r in self.responses if r.fallback_used)
        
        mood_distribution = {}
        topic_distribution = {}
        
        for response in self.responses:
            mood_distribution[response.mood] = mood_distribution.get(response.mood, 0) + 1
            topic_distribution[response.topic] = topic_distribution.get(response.topic, 0) + 1
        
        logging.info(f"""
        =================== SUMMARY REPORT ===================
        Total Interactions: {total_interactions}
        Unique Players: {len(players)}
        Fallback Responses Used: {fallback_count} ({fallback_count/total_interactions*100:.1f}%)
        
        Mood Distribution:
        {json.dumps(mood_distribution, indent=2)}
        
        Topic Distribution:
        {json.dumps(topic_distribution, indent=2)}
        
        Player States Summary:
        """)
        
        for player_id, state in self.player_states.items():
            logging.info(f"Player {player_id}: {state.interaction_count} interactions, current mood: {state.mood}")

def main():
    # Initialize system with Groq API key (set your API key in environment variable)
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        logging.warning("GROQ_API_KEY not found in environment variables. Using fallback only.")
    
    npc_system = NPCChatSystem(groq_api_key)
    
    # Load messages from JSON file
    messages = npc_system.load_messages('players.json')
    if not messages:
        logging.error("No messages loaded. Please check your players.json file.")
        return
    
    # Process all messages
    npc_system.process_all_messages(messages)
    
    # Save results
    npc_system.save_results()
    
    # Generate summary report
    npc_system.generate_summary_report()

if __name__ == "__main__":
    main()