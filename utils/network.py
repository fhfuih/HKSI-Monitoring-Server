import os


ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun.relay.metered.ca:80"]},
    {
        "urls": [
            "turn:global.relay.metered.ca:80",
            "turn:global.relay.metered.ca:80?transport=tcp",
            "turn:global.relay.metered.ca:443",
            "turns:global.relay.metered.ca:443?transport=tcp",
        ],
        "username": os.getenv("METERED_USERNAME"),
        "credential": os.getenv("METERED_CREDENTIAL"),
    },
]

END_SESSION_MESSAGE = "end session"
