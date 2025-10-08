import asyncio
from wyoming.client import AsyncClient
from wyoming.tts import Synthesize

async def test_tts():
    async with AsyncClient.connect("tcp://server:10600") as client:
        await client.write_event(Synthesize(text="Testing the Wyoming TTS").event())
        async for event in client.read_events():
            print(event)

asyncio.run(test_tts())