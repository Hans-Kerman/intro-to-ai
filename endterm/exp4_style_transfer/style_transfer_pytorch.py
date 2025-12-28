import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import datetime
import os
import csv
import time
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2): self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7): self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12): self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21): self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30): self.slice5.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x, layers):
        features = {}
        x = self.slice1(x)
        if 'relu1_1' in layers: features['relu1_1'] = x
        x = self.slice2(x)
        if 'relu2_1' in layers: features['relu2_1'] = x
        x = self.slice3(x)
        if 'relu3_1' in layers: features['relu3_1'] = x
        x = self.slice4(x)
        if 'relu4_1' in layers: features['relu4_1'] = x
        if 'relu4_2' in layers: features['relu4_2'] = x
        x = self.slice5(x)
        if 'relu5_1' in layers: features['relu5_1'] = x
        return features

class StyleLoss(nn.Module):
    def forward(self, input, target):
        G_input = self.gram_matrix(input)
        G_target = self.gram_matrix(target)
        return nn.MSELoss()(G_input, G_target)

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class ContentLoss(nn.Module):
    def forward(self, input, target):
        return nn.MSELoss()(input, target)

def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_width = int(image.size[0] * ratio)
        new_height = int(image.size[1] * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    elif shape:
        image = image.resize(shape, Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, original_size

def save_image_tensor(tensor, filename, original_size=None):
    image = tensor.clone().detach().cpu().squeeze(0)
    for t, m, s in zip(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)

    image = torch.clamp(image, 0, 1)
    image = transforms.ToPILImage()(image)

    if original_size:
        image = image.resize(original_size, Image.LANCZOS)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        image.save(filename)
    except PermissionError:
        print(f"[Warning] 无法保存图片到 {filename}，权限错误。")

def get_timestamp():
    return datetime.now().strftime("%m%d_%H%M%S")

def add_noise(content, noise_factor=0.00):
    return content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, default='weinisi.jpg')
    parser.add_argument('--style', type=str, default='style.jpg')
    parser.add_argument('--steps', type=int, default=50001)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--style_weight', type=float, default=1e6)
    parser.add_argument('--content_weight', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    EXP_NAME = f"exp_{get_timestamp()}"
    output_dir = f'output/{EXP_NAME}'
    os.makedirs('losses', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'checkpoints/{EXP_NAME}', exist_ok=True)

    CONTENT_LAYERS = ['relu4_2']
    STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    MAX_SIZE = args.size
    STYLE_WEIGHT = args.style_weight
    CONTENT_WEIGHT = args.content_weight
    TOTAL_STEPS = args.steps
    LEARNING_RATE = args.lr

    print("🔧 Starting style transfer with parameters:")
    print(f"📷 Content: {args.content} | 🎨 Style: {args.style}")
    print(f"📏 Max size: {MAX_SIZE} | ⏱️ Steps: {TOTAL_STEPS}")
    print(f"🎭 Style weight: {STYLE_WEIGHT} | 🧱 Content weight: {CONTENT_WEIGHT}")

    vgg = VGG19().to(device)
    style_loss = StyleLoss().to(device)
    content_loss = ContentLoss().to(device)

    content_img, content_size = load_image(args.content, max_size=MAX_SIZE)
    style_img, _ = load_image(args.style, max_size=MAX_SIZE)
    input_img = add_noise(content_img.clone(), noise_factor=0.00)
    input_img.requires_grad_(True)

    content_features = vgg(content_img, CONTENT_LAYERS)
    style_features = vgg(style_img, STYLE_LAYERS)

    optimizer = optim.Adam([input_img], lr=LEARNING_RATE)

    csv_path = f'checkpoints/{EXP_NAME}/loss_log.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'Content Loss', 'Style Loss', 'Total Loss', 'Time (s)'])

    best_total_loss = float('inf')
    best_img = None

    for step in range(TOTAL_STEPS):
        step_start = time.time()

        input_features = vgg(input_img, CONTENT_LAYERS + STYLE_LAYERS)

        content_loss_value = sum(content_loss(input_features[l], content_features[l]) for l in CONTENT_LAYERS)
        style_loss_value = sum(style_loss(input_features[l], style_features[l]) for l in STYLE_LAYERS)
        total_loss = CONTENT_WEIGHT * content_loss_value + STYLE_WEIGHT * style_loss_value

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        step_time = time.time() - step_start

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"❌ Loss is NaN/Inf at step {step}, stopping.")
            break

        if step % 100 == 0:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    content_loss_value.item(),
                    style_loss_value.item(),
                    total_loss.item(),
                    round(step_time, 4)
                ])
            save_image_tensor(input_img, f'{output_dir}/step_{step}.jpg', content_size)

        if total_loss < best_total_loss:
            best_total_loss = total_loss
            best_img = input_img.clone().detach()
            save_image_tensor(best_img, f'{output_dir}/best_total_loss.jpg', content_size)

        if step % 500 == 0:
            print(f'[{step}/{TOTAL_STEPS}] 🎯 Content: {content_loss_value.item():.4f}, '
                  f'Style: {style_loss_value.item():.4f}, Total: {total_loss.item():.4f}')

    save_image_tensor(input_img, f'{output_dir}/final_result.jpg', content_size)
    print(f'✅ Finished! Best Total Loss: {best_total_loss.item():.4f}')
    print(f'📁 Results saved in: {output_dir}')
