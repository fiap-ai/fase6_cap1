import os
import glob
from tqdm import tqdm

def remove_avif_files(directory):
    """Remove arquivos AVIF de um diretório."""
    # Encontrar todos os arquivos AVIF no diretório
    avif_files = glob.glob(os.path.join(directory, '*.avif'))
    
    if not avif_files:
        print(f"Nenhum arquivo AVIF encontrado em {directory}")
        return []
    
    print(f"Removendo {len(avif_files)} arquivos AVIF de {directory}...")
    
    removed_files = []
    for avif_path in tqdm(avif_files):
        try:
            # Remover o arquivo AVIF
            os.remove(avif_path)
            removed_files.append(avif_path)
        except Exception as e:
            print(f"Erro ao remover {avif_path}: {e}")
    
    print(f"Remoção concluída. {len(removed_files)} arquivos AVIF removidos com sucesso.")
    return removed_files

def main():
    # Diretórios do dataset
    dataset_dirs = [
        'dataset/train/images',
        'dataset/val/images',
        'dataset/test/images'
    ]
    
    total_removed = 0
    
    # Remover arquivos AVIF de cada diretório
    for directory in dataset_dirs:
        removed = remove_avif_files(directory)
        total_removed += len(removed)
    
    print(f"\nTotal de arquivos AVIF removidos: {total_removed}")
    print("Agora você pode usar o YOLOv5 com as imagens JPG.")

if __name__ == "__main__":
    main()
