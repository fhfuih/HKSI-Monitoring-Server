#!/usr/bin/env python

from io import BytesIO
from datetime import datetime
import asyncio
import os
from typing import Any

from dotenv import load_dotenv
import socketio
from PIL import Image, ImageDraw

sio = socketio.AsyncClient(ssl_verify=False)


@sio.event
async def connect():
    print("connection established")


@sio.event
async def prediction(data):
    print("prediction received with ", data)


@sio.event
async def disconnect():
    print("disconnected from server")


@sio.on("*")
async def any_event(event: str, sid: str, data: Any):
    print(f"Unregistered event {event} received with data {data} from {sid}")


def on_submit_frame(response):
    print("submit frame response", response)


def makeTime():
    t = datetime.now()
    ts = int(t.timestamp() * 1000)
    return ts.to_bytes(8, byteorder="little", signed=False)


def makeData(i):
    tsBytes = makeTime()

    imageBytesIO = BytesIO()
    image = Image.new("RGB", (244, 244), "#B99169")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), str(i), fill="black")
    image.save(imageBytesIO, format="PNG")
    imageBytes = imageBytesIO.getvalue()
    imageBytesIO.close()

    data = tsBytes + imageBytes
    return data


async def send():
    print("Submit a frame before session_start")
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await sio.emit("frame", makeData(1), callback=lambda *args: future.set_result(args))
    await future
    print("Frame response is", future.result())

    print("Send session_start")
    future = loop.create_future()
    await sio.emit(
        "session_start", makeTime(), callback=lambda *args: future.set_result(args)
    )
    await future
    print("session_start response is", future.result())

    print("Submit frames")
    await sio.emit("frame", makeData(1), callback=on_submit_frame)
    await asyncio.sleep(0.3)
    await sio.emit("frame", makeData(2), callback=on_submit_frame)
    await asyncio.sleep(0.3)
    await sio.emit("frame", makeData(3), callback=on_submit_frame)
    await asyncio.sleep(0.3)
    #
    await asyncio.sleep(0.3)

    print("Sending session_end")
    await sio.emit(
        "session_end",
        makeTime(),
        callback=lambda x: print("session_end response is", x),
    )


async def main():
    await sio.connect(
        "ws://127.0.0.1:8765",
        auth={"key": os.getenv("SOCKETIO_PASSWORD", "")},
        transports=["websocket"],
    )
    await asyncio.gather(
        sio.wait(),
        send(),
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
