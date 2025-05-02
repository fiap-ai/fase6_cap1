import os
import glob
import cv2
import concurrent.futures
from tqdm import tqdm

def convert_image(avif_path):
    """Converte uma imagem AVIF para JPG usando OpenCV."""
    try:
        # Ler a imagem com OpenCV
        img = cv2.imread(avif_path)
        
        if img is None:
            print(f"Erro ao ler {avif_path} com OpenCV")
            return None
        
        # Criar o caminho para a imagem JPG (mesmo nome, extensão diferente)
        jpg_path = os.path.splitext(avif_path)[0] + '.jpg'
        
        # Salvar como JPG
        cv2.imwrite(jpg_path, img)
        
        return jpg_path
    except Exception as e:
        print(f"Erro ao converter {avif_path}: {e}")
        return None

def convert_directory(directory):
    """Converte todas as imagens AVIF em um diretório para JPG."""
    # Encontrar todas as imagens AVIF no diretório
    avif_files = glob.glob(os.path.join(directory, '*.avif'))
    
    if not avif_files:
        print(f"Nenhuma imagem AVIF encontrada em {directory}")
        return []
    
    print(f"Convertendo {len(avif_files)} imagens AVIF em {directory}...")
    
    # Converter as imagens em paralelo
    converted_files = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(convert_image, avif_files), total=len(avif_files)))
        
    # Filtrar resultados bem-sucedidos
    converted_files = [f for f in results if f is not None]
    
    print(f"Conversão concluída. {len(converted_files)} imagens convertidas com sucesso.")
    return converted_files

def main():
    # Diretórios do dataset
    dataset_dirs = [
        'dataset/train/images',
        'dataset/val/images',
        'dataset/test/images'
    ]
    
    total_converted = 0
    
    # Converter imagens em cada diretório
    for directory in dataset_dirs:
        converted = convert_directory(directory)
        total_converted += len(converted)
    
    print(f"\nTotal de imagens convertidas: {total_converted}")
    print("Agora você pode usar o YOLOv5 com as imagens JPG.")

if __name__ == "__main__":
    main()
