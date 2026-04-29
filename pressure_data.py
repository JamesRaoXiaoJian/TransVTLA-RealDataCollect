import socket
import struct
import time
import csv
from datetime import datetime
from pathlib import Path

UDP_PORT = 4321
BUFFER_SIZE = 136  # 修改为新数据包长度
PACKET_FORMAT = "<Q64h"  # uint64 + 64个 int16_t
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
BATCH_SIZE = 100
FLUSH_INTERVAL_S = 0.1

# 发送握手包
esp32_ip = "192.168.31.164"  # ESP32的局域网IP
esp32_port = 2222  # 下位机监听端口

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))

# 关键修改：设置 socket 超时时间（例如 1.0 秒）
# 避免 recvfrom 永久阻塞，使 Python 能够响应 KeyboardInterrupt
sock.settimeout(1.0) 

# 向下位机发送一次握手包
sock.sendto(b"HELLO", (esp32_ip, esp32_port))
print(f"Sent handshake to {esp32_ip}:{esp32_port}")
time.sleep(0.5)

print(f"Listening on UDP port {UDP_PORT}...")

output_dir = Path("pressure_logs")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"pressure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(output_file, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp_us", "left", "right"])
print(f"Saving data to: {output_file}")
row_buffer: list[list[int]] = []
last_flush_time = time.time()
last_print_time = 0.0

try:
    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
        except socket.timeout:
            # 发生超时，直接进入下一次循环。
            # 此时解释器会检查是否有 Ctrl+C 信号产生。
            now = time.time()
            if row_buffer and (now - last_flush_time) >= FLUSH_INTERVAL_S:
                csv_writer.writerows(row_buffer)
                csv_file.flush()
                row_buffer.clear()
                last_flush_time = now
            continue
            
        if len(data) < PACKET_SIZE:
            print(f"[{addr}] Packet too small: {len(data)} bytes")
            continue
            
        timestamp_us, *values = struct.unpack(PACKET_FORMAT, data)
        left = values[63]   # CH64
        right = values[62]  # CH63

        now = time.time()
        if (now - last_print_time) >= 1.0:
            print(f"[{addr[0]}] Timestamp: {timestamp_us} us | left: {left} | right: {right}")
            last_print_time = now

        row_buffer.append([timestamp_us, left, right])

        if len(row_buffer) >= BATCH_SIZE or (now - last_flush_time) >= FLUSH_INTERVAL_S:
            csv_writer.writerows(row_buffer)
            csv_file.flush()
            row_buffer.clear()
            last_flush_time = now
            
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    if row_buffer:
        csv_writer.writerows(row_buffer)
        csv_file.flush()
    csv_file.close()
    sock.close()