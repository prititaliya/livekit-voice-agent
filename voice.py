from dataclasses import dataclass
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, function_tool, RunContext,function_tool
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel


load_dotenv(".env.local")

@dataclass
class MySessionInfo:
    user_name: str = None
    age: int = None
    parent_name: str = None
    is_consented: bool = None

class IntakeAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are an intake agent. Learn the user's name and age. donnot reveal that you are an AI model and any model information. and if asked denie giving any such information."
        )

    @function_tool()
    async def record_name(self, context: RunContext[MySessionInfo], name: str):
        context.userdata.user_name = name
        return self._handoff_if_done(context)

    @function_tool()
    async def record_age(self, context: RunContext[MySessionInfo], age: int):
        context.userdata.age = age
        return self._handoff_if_done(context)

    def _handoff_if_done(self, context):
        if context.userdata.user_name and context.userdata.age:
            if context.userdata.age >= 18:
                context.userdata.is_consented = True
                return CustomerServiceAgent()
            else:
                return ParentalConsentAgent()
        return None
  
    
class ParentalConsentAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="Parental consent is not given. Politely inform the user that they must have parental consent to proceed since they are under 18. donnot reveal that you are an AI model and any model information and if asked denie giving any such information. AND DONOT GIVE ANY ANSWER UNTILL YOU HAVE THE PARENTAL CONSENT. STRICLTY ASK FOR THE PARENTAL CONSENT AND THE PARENT'S NAME.",
            tts="elevenlabs/eleven_flash_v2_5"
        )


    @function_tool()
    async def record_parental_consent(self, context: RunContext[MySessionInfo], parent_name:str):
        context.userdata.parent_name = parent_name
        return self._handoff_if_done(context)

    async def on_enter(self) -> None:
        userdata: MySessionInfo = self.session.userdata
        await self.session.generate_reply(
            instructions=f"ask {userdata.user_name} for parental consent and the parent's name."
        )
    @function_tool()
    async def record_consent(self, context: RunContext[MySessionInfo], is_consented: bool):
        context.userdata.is_consented = is_consented
        self._handoff_if_done(context)

    def _handoff_if_done(self, context: RunContext[MySessionInfo]):
        if context.userdata.is_consented:
            return CustomerServiceAgent()
        else:
            return None


class CustomerServiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a friendly customer service representative. Donnot reveal that you are an AI model and any model information and if asked denie giving any such information. DONNOT GIVE ANY NUTRITIONAL INFORMATION WITHOUT USING THE get_nutritional_info TOOL. USE INFORMATION FROM TOOLS ONLY TO ANSWER USER QUERIES ABOUT NUTRITIONAL INFORMATION.",
                         tts="elevenlabs/eleven_flash_v2_5")

    @function_tool(description="Get the current date and time.")
    async def get_current_datetime(
        self,
        context: RunContext, 
    ) -> str:
        """Get the current date and time as a formatted string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @function_tool(description="Get the current weather for a given city.")
    async def get_current_weather(
        self,
        context: RunContext, 
        city: str,
    ) -> str:
        import requests
        import os
        
        api_key = os.getenv("OPEN_WEATHER_API_KEY")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data["cod"] != 200:
            return f"Could not retrieve weather data for {city}."
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"The current weather in {city} is {weather_desc} with a temperature of {temp} Celsius."
    
    @function_tool(description="Get nutritional information for a given product.")
    async def get_nutritional_info(
        self,
        context: RunContext, 
        product_name: str,  ) -> str:
        import requests
        import os
        from dotenv import load_dotenv

        load_dotenv('.env.local')

        url = "https://world.openfoodfacts.org/api/v2/search"
        headers = {
            "User-Agent": "healme (prititaliya2244@gmail.com)",
            "Accept": "application/json"
        }

        params = {
            "categories_tags": product_name,      
            "fields": "nutriments"
        }

        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        nutritional_info = data.get("products", [{}])[0].get("nutriments", {})
        print(nutritional_info)

        return str(nutritional_info)
    async def on_enter(self) -> None:
        userdata: MySessionInfo = self.session.userdata
        await self.session.generate_reply(
            instructions=f"Greet {userdata.user_name} personally and offer your assistance."
        )


async def entrypoint(ctx: agents.JobContext):
 
    session = AgentSession(
        preemptive_generation=True,
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    session.userdata = MySessionInfo()

    await session.start(
        room=ctx.room,
        agent=CustomerServiceAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


    await session.generate_reply(
        instructions="Introduce yourself and ask the user for their name and age."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
