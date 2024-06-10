#!/usr/bin/env python

from io import BytesIO
from datetime import datetime
import asyncio
import websockets
from PIL import Image, ImageDraw

ws = None


def makeData():
    t = datetime.now()
    ts = int(t.timestamp() * 1000)

    imageBytesIO = BytesIO()
    image = Image.new("RGB", (244, 244), "#B99169")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), f"Time: {t}", fill="black")
    image.save(imageBytesIO, format="PNG")
    imageBytes = imageBytesIO.getvalue()
    imageBytesIO.close()

    data = ts.to_bytes(8, byteorder="little", signed=False) + imageBytes
    return data, t


async def send():
    assert ws is not None
    print("Sending data")
    for i in range(5):
        print(f"Making data #{i}")
        data, t = makeData()
        print(f"Sending at {t} with data length {len(data)} bytes")
        await ws.send(data)
        await asyncio.sleep(0.4)


async def handle_response():
    assert ws is not None
    print("Listening for responses")
    i = 0
    async for message in ws:
        print(f"Received: {message}")
        i += 1
        if i == 5:
            break


async def main():
    global ws
    ws = await websockets.connect("ws://localhost:8765")
    print("ws initialized")
    await asyncio.gather(
        handle_response(),
        send(),
    )
    await ws.close()


asyncio.run(main())
