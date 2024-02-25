import socket
import struct
import pipeline
import paths

def receive_frame(client_socket, frame_size):
    frame_data = b""
    bytes_received = 0

    print("frame size: ", frame_size)


    while bytes_received < frame_size:
        try:
            chunk = client_socket.recv(min(frame_size - bytes_received, 8388608))
            if not chunk:
                break
            frame_data += chunk
            bytes_received += len(chunk)

        except:
            return frame_data

    return frame_data

def format_model_output(output_string):
    flat_list = [float(i) for i in output_string.split(',')]
    keypoints_3d = [flat_list[i:i+3] for i in range(0, len(flat_list), 3)]
    formatted_string = str(keypoints_3d)
    
    return formatted_string

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)

    print('Waiting for client connection...')
    client_socket, address = server_socket.accept()
    print('Client connected:', address)

    while True:
       
        size_data = client_socket.recv(4)
        frame_size = struct.unpack('!I', size_data)[0]

        # Receive the frame from the client
        frame_data = receive_frame(client_socket, frame_size)

        # Process the received frame as needed
        with open(image_path, 'wb') as f:
            f.write(frame_data)

        print('Frame received successfully!')

        try:
            outputstring3d = pipeline.simple_yolo(yolov8_model_path, sye_model_path, image_path)
            print("Processing complete, output:", outputstring3d)

            formatted_outputstring3d = format_model_output(outputstring3d)
            print("Formatted output for sending:", formatted_outputstring3d)

            client_socket.sendall(str(formatted_outputstring3d).encode())
        except Exception as e:
            print(f"Error processing image: {e}")

            no_detection_message = "no detection"
            client_socket.sendall(no_detection_message.encode())

    
    client_socket.close()
    server_socket.close()

server_ip = ''  # listen from any ip
server_port = 6000  
image_path = paths.image_path
yolov8_model_path = paths.yolov8_model_path
sye_model_path = paths.sye_model_path
main()