from dataclasses import dataclass
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, function_tool, RunContext
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
        super().__init__(instructions="You are a friendly customer service representative. Donnot reveal that you are an AI model and any model information and if asked denie giving any such information.",
                         tts="elevenlabs/eleven_flash_v2_5")

    async def on_enter(self) -> None:
        userdata: MySessionInfo = self.session.userdata
        await self.session.generate_reply(
            instructions=f"Greet {userdata.user_name} personally and offer your assistance."
        )


async def entrypoint(ctx: agents.JobContext):
    """Start a LiveKit AgentSession configured for voice with intake->CSR handoff.

    Run this file with the LiveKit worker CLI runner (see bottom of file).
    """
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
        agent=IntakeAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


    await session.generate_reply(
        instructions="Introduce yourself and ask the user for their name and age."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
