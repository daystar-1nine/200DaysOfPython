# TCP Three-Way Handshake and Connection Teardown

The Transmission Control Protocol (TCP) establishes reliable bidirectional connections using a three-way handshake:
1. SYN: The client generates an Initial Sequence Number (ISN) and sends a packet with the SYN flag set.
2. SYN-ACK: The server responds with its own SYN packet and acknowledges the client's ISN with an ACK (ISN + 1).
3. ACK: The client acknowledges the server's sequence number with an ACK packet. The connection enters the ESTABLISHED state.

Connection teardown uses a four-way handshake involving FIN and ACK packets from both endpoints, with the initiating client entering the TIME_WAIT state to ensure duplicate segments expire.