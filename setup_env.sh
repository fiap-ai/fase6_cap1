#!/bin/bash
# Script para configuração do ambiente de desenvolvimento para o projeto de visão computacional

echo "Configurando ambiente para o projeto de visão computacional..."

# Verificar se o Python está instalado
if command -v python3 &>/dev/null; then
    echo "Python 3 encontrado."
else
    echo "Python 3 não encontrado. Por favor, instale o Python 3 antes de continuar."
    exit 1
fi

# Verificar se já estamos em um ambiente virtual
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Já estamos em um ambiente virtual: $VIRTUAL_ENV"
    echo "Usando o ambiente virtual atual."
else
    # Criar ambiente virtual
    echo "Criando ambiente virtual..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Ambiente virtual 'venv' ativado."
fi

# Instalar dependências
echo "Instalando dependências..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install numpy
pip install pandas
pip install scikit-learn
pip install jupyter
pip install ipykernel
pip install seaborn
pip install tqdm
pip install Pillow
pip install pillow-avif-plugin  # Adicionado para suporte a imagens AVIF

# Instalar YOLOv5
echo "Clonando repositório YOLOv5..."
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..

# Configurar kernel Jupyter
echo "Configurando kernel Jupyter..."
python -m ipykernel install --user --name=venv_vision --display-name="Python (Vision)"

# Criar estrutura de diretórios
# echo "Criando estrutura de diretórios..."
# mkdir -p notebooks
# mkdir -p data/raw
# mkdir -p data/processed
# mkdir -p models/yolo
# mkdir -p models/cnn
# mkdir -p results/yolo_custom
# mkdir -p results/yolo_traditional
# mkdir -p results/cnn

echo "Ambiente configurado com sucesso!"
echo "Para ativar o ambiente, execute: source venv/bin/activate"
echo "Para iniciar o Jupyter Notebook, execute: jupyter notebook"
