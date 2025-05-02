"""
Funções utilitárias para o projeto de visão computacional.
"""

def get_project_dir():
    """Retorna o diretório do projeto."""
    return "."

def setup_environment():
    """Verifica o ambiente para os notebooks subsequentes."""
    # Definir o diretório do projeto
    project_dir = "."
    
    return project_dir, False

def visualize_image(image_path, figsize=(10, 10)):
    """Visualiza uma imagem do dataset."""
    import matplotlib.pyplot as plt
    import cv2
    
    # Carregar a imagem
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter de BGR para RGB
    
    # Mostrar a imagem
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    return img

def visualize_image_with_bbox(image_path, label_path, class_names, figsize=(10, 10)):
    """Visualiza uma imagem com suas bounding boxes."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    import numpy as np
    
    # Carregar a imagem
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter de BGR para RGB
    height, width, _ = img.shape
    
    # Carregar as anotações
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Mostrar a imagem
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    
    # Adicionar bounding boxes
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        bbox_width = float(parts[3]) * width
        bbox_height = float(parts[4]) * height
        
        # Calcular coordenadas do canto superior esquerdo
        x1 = x_center - bbox_width / 2
        y1 = y_center - bbox_height / 2
        
        # Adicionar retângulo
        rect = patches.Rectangle((x1, y1), bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Adicionar rótulo
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        ax.text(x1, y1, class_name, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.axis('off')
    plt.show()
    
    return img
