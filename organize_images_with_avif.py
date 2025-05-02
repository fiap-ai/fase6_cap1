import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import subprocess

# Definir caminhos
apple_src_dir = 'assets/apple'
banana_src_dir = 'assets/banana'
train_dir = 'dataset/train/images'
val_dir = 'dataset/val/images'
test_dir = 'dataset/test/images'
train_labels_dir = 'dataset/train/labels'
val_labels_dir = 'dataset/val/labels'
test_labels_dir = 'dataset/test/labels'

# Garantir que as pastas de destino existam
for directory in [train_dir, val_dir, test_dir, train_labels_dir, val_labels_dir, test_labels_dir]:
    os.makedirs(directory, exist_ok=True)

def convert_avif_to_jpg(avif_path, jpg_path):
    """Converte uma imagem AVIF para JPG usando ffmpeg."""
    try:
        # Usar ffmpeg para converter AVIF para JPG
        cmd = ['ffmpeg', '-i', avif_path, '-y', jpg_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao converter {avif_path} com ffmpeg: {e}")
        return False
    except Exception as e:
        print(f"Erro ao converter {avif_path}: {e}")
        return False

# Função para organizar as imagens
def organize_images(src_dir, category, train_count=32, val_count=4, test_count=4):
    # Listar todas as imagens na pasta de origem
    images = [f for f in os.listdir(src_dir) if f.endswith('.avif')]
    
    # Embaralhar as imagens para garantir aleatoriedade
    random.shuffle(images)
    
    # Selecionar imagens para cada conjunto
    train_images = images[:train_count]
    val_images = images[train_count:train_count+val_count]
    test_images = images[train_count+val_count:train_count+val_count+test_count]
    
    # Função para converter e salvar imagem
    def convert_and_save(src_path, dst_path, label_path, class_id):
        try:
            # Converter AVIF para JPG
            if convert_avif_to_jpg(src_path, dst_path):
                # Criar arquivo de rótulo (label)
                # Formato YOLO: <class_id> <x_center> <y_center> <width> <height>
                # Para simplificar, vamos assumir que o objeto ocupa o centro da imagem com 50% de largura/altura
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.5 0.5")
                return True
            return False
        except Exception as e:
            print(f"Erro ao processar {src_path}: {e}")
            return False
    
    # Processar imagens de treino
    print(f"Processando {len(train_images)} imagens para treino...")
    for i, img in enumerate(tqdm(train_images)):
        src_path = os.path.join(src_dir, img)
        new_name = f"{category}_{i+1:03d}"
        dst_path = os.path.join(train_dir, f"{new_name}.jpg")
        label_path = os.path.join(train_labels_dir, f"{new_name}.txt")
        convert_and_save(src_path, dst_path, label_path, 0 if category == 'apple' else 1)
    
    # Processar imagens de validação
    print(f"Processando {len(val_images)} imagens para validação...")
    for i, img in enumerate(tqdm(val_images)):
        src_path = os.path.join(src_dir, img)
        new_name = f"{category}_val_{i+1:03d}"
        dst_path = os.path.join(val_dir, f"{new_name}.jpg")
        label_path = os.path.join(val_labels_dir, f"{new_name}.txt")
        convert_and_save(src_path, dst_path, label_path, 0 if category == 'apple' else 1)
    
    # Processar imagens de teste
    print(f"Processando {len(test_images)} imagens para teste...")
    for i, img in enumerate(tqdm(test_images)):
        src_path = os.path.join(src_dir, img)
        new_name = f"{category}_test_{i+1:03d}"
        dst_path = os.path.join(test_dir, f"{new_name}.jpg")
        label_path = os.path.join(test_labels_dir, f"{new_name}.txt")
        convert_and_save(src_path, dst_path, label_path, 0 if category == 'apple' else 1)

def main():
    # Verificar se ffmpeg está instalado
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpeg está instalado. Prosseguindo com a conversão...")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERRO: ffmpeg não está instalado. Por favor, instale o ffmpeg antes de continuar.")
        print("No macOS: brew install ffmpeg")
        print("No Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("No Windows: https://ffmpeg.org/download.html")
        return
    
    # Limpar os diretórios de destino
    print("Limpando diretórios de destino...")
    for directory in [train_dir, val_dir, test_dir]:
        for file in os.listdir(directory):
            if file.endswith('.jpg'):
                os.remove(os.path.join(directory, file))
    
    for directory in [train_labels_dir, val_labels_dir, test_labels_dir]:
        for file in os.listdir(directory):
            if file.endswith('.txt'):
                os.remove(os.path.join(directory, file))
    
    # Organizar imagens de maçãs
    print("Organizando imagens de maçãs...")
    organize_images(apple_src_dir, 'apple')
    
    # Organizar imagens de bananas
    print("Organizando imagens de bananas...")
    organize_images(banana_src_dir, 'banana')
    
    print("Organização concluída!")

if __name__ == "__main__":
    main()
