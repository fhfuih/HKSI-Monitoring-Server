from datetime import datetime
from io import BytesIO
import asyncio
import json

from websockets.server import serve, WebSocketServerProtocol
from PIL import Image

import mock_model

# WebSocket server address
HOST = "localhost"
PORT = 8765

# list of prediction functions to run (e.g., HR is 1, Fatigue is 2)
PREDICTIONS = [
    mock_model.foward_single_frame_1,
    mock_model.foward_single_frame_2,
]


async def process_frame(websocket: WebSocketServerProtocol):
    async for message in websocket:
        if isinstance(message, str):
            continue

        # extract the first 64 bits of `message` as an unsigned integer
        timestamp = message[:8]
        timestamp = int.from_bytes(timestamp, byteorder="little", signed=False)
        t = datetime.fromtimestamp(timestamp / 1000)
        print(f"Received data at {t} with length {len(message)} bytes")

        # extract the remaining bytes of `message` as a png image
        frame = message[8:]
        frameIO = BytesIO(frame)
        frame = Image.open(frameIO, formats=("PNG",))
        frameIO.close()

        # run prediction and return
        results = await asyncio.gather(
            *[
                asyncio.to_thread(prediction_func, frame, timestamp)
                for prediction_func in PREDICTIONS
            ]
        )
        result = {
            "frame": {"size": frame.size, "mode": frame.mode},
            "recv_ts": timestamp,
        }
        for r in results:
            if r is not None:
                result.update(r)
        print(f"Responding data at {t} with {result}")
        await websocket.send(json.dumps(result, ensure_ascii=False))


async def main():
    async with serve(process_frame, HOST, PORT):
        print(f"Server started at ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


asyncio.run(main())
