import os
import torch
import numpy as np
from PIL import Image
dtype = torch.FloatTensor

def read_yuv_video(yuv_filename, width, height, frame_count, color_format='420'):
    if color_format == '420':
        frame_size = (width * height) + (width // 2 * height // 2 * 2)
    with open(yuv_filename, 'rb') as f:
        raw_data = f.read()
    video_data = np.zeros((height, width, frame_count), dtype=np.float32)
    # 逐帧读取并解析 YUV 数据
    for frame_idx in range(frame_count):
        start_idx = frame_idx * frame_size
        end_idx = start_idx + frame_size
        # 读取当前帧的 YUV 数据
        frame_data = raw_data[start_idx:end_idx]
        # 提取 Y, U, V 分量
        y_data = frame_data[:width * height]
        uv_data = frame_data[width * height:]
        # 将 Y 分量转换为二维数组
        y_frame = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)
        # 将 U, V 分量转换为二维数组
        # if color_format == '420':
        #     u_data = uv_data[:width * height // 4]
        #     v_data = uv_data[width * height // 4:]
        #     u_frame = np.frombuffer(u_data, dtype=np.uint8).reshape(height // 2, width // 2)
        #     v_frame = np.frombuffer(v_data, dtype=np.uint8).reshape(height // 2, width // 2)
        #     u_frame = np.repeat(np.repeat(u_frame, 2, axis=0), 2, axis=1)
        #     v_frame = np.repeat(np.repeat(v_frame, 2, axis=0), 2, axis=1)
        # 将 Y, U, V 分量合并为 RGB 图像（可选）
        # 你可以使用 OpenCV 或其他库将 YUV 转换为 RGB 但是这里只保留 Y 分量
        video_data[:, :, frame_idx] = y_frame / 255.0
    return video_data

def generate_random_mask_3d(H, W, C, visible_ratio=0.1):
    num_visible = int(H * W * C * visible_ratio)
    all_positions = np.arange(H * W * C)
    visible_positions = np.random.choice(all_positions, size=num_visible, replace=False)
    mask = np.zeros(H * W * C, dtype=np.uint8)
    mask[visible_positions] = 1
    mask = mask.reshape(H, W, C)
    mask = torch.from_numpy(mask).type(dtype)
    return mask

def generate_random_mask(H, W, visible_ratio=0.1):
    num_visible = int(H * W * visible_ratio)
    all_positions = np.arange(H * W)
    visible_positions = np.random.choice(all_positions, size=num_visible, replace=False)
    mask = np.zeros(H * W, dtype=np.uint8)
    mask[visible_positions] = 1
    mask = mask.reshape(H, W, 1)
    mask = np.repeat(mask, 3, axis=2)
    mask = torch.from_numpy(mask).type(dtype)
    return mask

def load_grayscale_images_from_directory(directory):
    png_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    images = []
    for file in png_files:
        file_path = os.path.join(directory, file)
        image = Image.open(file_path)  # 读取为灰度图像
        image_array = np.array(image)  # 转换为 NumPy 数组
        images.append(image_array)
    if len(images) > 0:
        max_value = 2 ** 16 - 1
        stacked_images = np.stack(images, axis=-1).astype(np.float32) / max_value
        return stacked_images
    else:
        return None

def truncated_linear_stretch(image, truncated_value=0, maxout=1, minout=0):
    def gray_process(gray, maxout, minout):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray_new = (gray - truncated_down) / ((truncated_up - truncated_down) / (maxout - minout))
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return np.float32(gray_new)

    height, width, band = image.shape
    out =np.zeros((height, width, band))
    
    for b in range(band):
        out[:,:,b] = gray_process(image[:,:,b], maxout, minout)
    return out