#!/usr/bin/env python

from io import BytesIO
from datetime import datetime
import asyncio
import json
from typing import Any

import socketio
from PIL import Image, ImageDraw

sio = socketio.AsyncClient()


@sio.event
async def connect():
    print("connection established")


@sio.event
async def prediction(data):
    print("message received with ", data)
    await sio.emit("my response", {"response": "my response"})


@sio.event
async def disconnect():
    print("disconnected from server")


@sio.on("*")
async def any_event(event: str, sid: str, data: Any):
    print(f"Event {event} received with data {data} from {sid}")


def on_prediction(response):
    print("message received with", json.loads(response) if response else "empty")


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
    print("Sending data")
    for i in range(5):
        print(f"Making data #{i}")
        data, t = makeData()
        print(f"Sending at {t} with data length {len(data)} bytes")
        await sio.emit("frame", data, callback=on_prediction)
        await asyncio.sleep(0.4)


async def main():
    await sio.connect("http://localhost:8765")
    await asyncio.gather(
        sio.wait(),
        send(),
    )


if __name__ == "__main__":
    asyncio.run(main())
