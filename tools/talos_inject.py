import socket
import struct
import sys
import argparse

NERVE_SOCKET = "/tmp/talos_nerve.sock"

def recv_all(sock, n):
    """Helper to receive exactly n bytes."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def send_thought(text):
    if not text: return
    
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(NERVE_SOCKET)
        
        # 1. Send Request [Len][Payload]
        payload = text.encode('utf-8')
        length = struct.pack("!I", len(payload))
        client.sendall(length + payload)
        
        # 2. Receive Response [Len][Payload]
        len_bytes = recv_all(client, 4)
        if not len_bytes:
            print("[!] No response from Talos.")
            return
            
        resp_len = struct.unpack("!I", len_bytes)[0]
        resp_bytes = recv_all(client, resp_len)
        
        print(f"\n[TALOS]: {resp_bytes.decode('utf-8')}\n")
        
    except FileNotFoundError:
        print("[!] Talos is not listening. Restart the daemon.")
    except Exception as e:
        print(f"[!] Connection Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talos-O Neural Interface Injector")
    parser.add_argument("input", nargs="?", help="Text string to inject")
    parser.add_argument("-f", "--file", help="Inject content of a file")
    
    args = parser.parse_args()
    
    content = ""
    if args.file:
        try:
            with open(args.file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"[!] Error reading file: {e}")
            sys.exit(1)
    elif args.input:
        content = args.input
    else:
        print("[!] Usage: python talos_inject.py \"My thought\" OR -f filename.txt")
        sys.exit(1)

    if content:
        print(f"[*] Injecting {len(content)} bytes into Cognitive Plane...")
        send_thought(content)
