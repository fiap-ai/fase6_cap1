# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# FarmTech Solutions - Sistema de Visão Computacional

## 🔗 Links Importantes
- [Notebook Completo do Projeto](notebooks/GabrielMule_RM560586_fase6.ipynb)
- [Notebook 1: Treinamento YOLO Customizado (30 épocas)](notebooks/01_treinamento_yolo_custom.ipynb)
- [Notebook 2: Treinamento YOLO Customizado (60 épocas)](notebooks/02_treinamento_yolo_custom_60epocas.ipynb)
- [Notebook 3: Validação e Teste](notebooks/03_validacao_teste.ipynb)
- [Notebook 4: YOLO Tradicional](notebooks/04_yolo_tradicional.ipynb)
- [Notebook 5: CNN do Zero (Parte 1)](notebooks/05_cnn_do_zero_parte1.ipynb)
- [Notebook 6: CNN do Zero (Parte 2)](notebooks/06_cnn_do_zero_parte2.ipynb)
- [Notebook 7: CNN do Zero (Parte 3)](notebooks/07_cnn_do_zero_parte3.ipynb)
- [Notebook 8: Comparação Final](notebooks/08_comparacao_final.ipynb)
- [Vídeo Demonstrativo](https://youtu.be/XXXXXXXX) <!-- Substitua pelo link do vídeo -->

## 👨‍🎓 Integrantes: 
- <a href="https://www.linkedin.com/in/XXXXXXXX/">Gabriel Mule</a> - RM 560586

## 👩‍🏫 Professores:
### Tutor(a) 
- <a href="https://www.linkedin.com/in/XXXXXXXX">Nome do Professor</a>

## 📜 Descrição

Este projeto implementa e compara diferentes abordagens de visão computacional para detecção e classificação de objetos, desenvolvido no contexto da FarmTech Solutions, uma empresa que está expandindo seus serviços de IA para além do agronegócio. O foco principal é demonstrar o potencial e a acurácia do YOLO (You Only Look Once) para detecção de objetos, comparando-o com outras abordagens.

O sistema implementa e avalia três diferentes métodos:

1. **YOLO Customizado**: Treinado especificamente para detectar duas categorias de objetos, com duas configurações diferentes (30 e 60 épocas)
2. **YOLO Tradicional**: Modelo pré-treinado aplicado às mesmas imagens
3. **CNN do Zero**: Implementação de uma rede neural convolucional para classificação

## 📊 Visão Computacional e Machine Learning

### Fase 1: Preparação do Ambiente e Dataset
- Configuração do ambiente de desenvolvimento
- Seleção e exploração de datasets públicos
- Escolha de duas categorias de objetos visualmente distintas
- Coleta de 40 imagens para cada categoria

### Fase 2: Organização do Dataset
- Divisão em conjuntos de treino (32 imagens), validação (4 imagens) e teste (4 imagens) por categoria
- Organização das imagens no Google Drive
- Estruturação de diretórios para facilitar o acesso durante o treinamento

### Fase 3: Rotulação de Dados
- Utilização do site Make Sense IA para rotulação das imagens
- Criação de anotações no formato adequado para YOLO
- Verificação da qualidade das anotações

### Fase 4: Treinamento do Modelo YOLO
- Implementação do YOLO customizado baseado na arquitetura YOLOv5
- Treinamento com duas configurações: 30 e 60 épocas
- Monitoramento de métricas como mAP, precision, recall e loss
- Ajuste de hiperparâmetros para otimizar o desempenho

### Fase 5: Validação e Teste
- Avaliação dos modelos usando o conjunto de teste
- Comparação de resultados entre as diferentes configurações
- Captura de prints das imagens de teste processadas
- Análise de métricas de desempenho

### Fase 6: YOLO Tradicional
- Aplicação do YOLO tradicional pré-treinado no dataset COCO
- Avaliação de desempenho nas categorias específicas do projeto
- Comparação com o YOLO customizado

### Fase 7: CNN do Zero
- Implementação de uma CNN personalizada para classificação
- Treinamento com técnicas de augmentação de dados e regularização
- Avaliação usando métricas de classificação

### Fase 8: Comparação de Modelos
- Análise comparativa de todas as abordagens
- Avaliação de facilidade de uso, precisão, tempo de treinamento e inferência
- Identificação de cenários de uso ideais para cada abordagem

## 📊 Resultados e Comparação

### Desempenho dos Modelos

| Modelo | mAP@0.5 | Precision | Recall | Tempo de Inferência |
|--------|---------|-----------|--------|---------------------|
| YOLO Customizado (30 épocas) | 0.54109 | 0.40501 | 0.625 | - |
| YOLO Customizado (60 épocas) | 0.76108 | 0.89753 | 0.625 | - |
| YOLO Tradicional | - | - | - | - |
| CNN do Zero | N/A | 0.96 (Acurácia) | 0.96 (Acurácia) | - |

### Análise Comparativa

1. **YOLO Customizado**:
   - **Pontos fortes**: Maior precisão para as categorias específicas, detecção otimizada
   - **Limitações**: Requer tempo e recursos para treinamento, necessita de dados rotulados

2. **YOLO Tradicional**:
   - **Pontos fortes**: Uso imediato sem treinamento, bom desempenho geral
   - **Limitações**: Menos preciso para categorias específicas, limitado às classes do COCO

3. **CNN do Zero**:
   - **Pontos fortes**: Mais simples e eficiente para classificação pura
   - **Limitações**: Não fornece informações de localização, dificuldade com múltiplos objetos

### Cenários de Uso Ideais

- **YOLO Customizado**: Quando precisão em categorias específicas é crítica e há recursos disponíveis
- **YOLO Tradicional**: Para prototipagem rápida ou quando as categorias de interesse estão no COCO
- **CNN do Zero**: Para classificação simples quando localização não é necessária

## 📺 Demonstração

O projeto pode ser testado através dos notebooks Jupyter, que demonstram:
- Organização e rotulação do dataset
- Treinamento e validação do YOLO customizado
- Aplicação do YOLO tradicional
- Implementação da CNN do zero
- Comparação entre as diferentes abordagens

### Vídeo Demonstrativo

[Assista ao vídeo demonstrativo do projeto](https://youtu.be/XXXXXXXX) <!-- Substitua pelo link do vídeo -->

## 📁 Estrutura de Arquivos

```
projeto/
├── notebooks/
│   ├── GabrielMule_RM560586_fase6.ipynb      # Notebook completo do projeto
│   ├── 01_treinamento_yolo_custom.ipynb      # Treinamento YOLO (30 épocas)
│   ├── 02_treinamento_yolo_custom_60epocas.ipynb # Treinamento YOLO (60 épocas)
│   ├── 03_validacao_teste.ipynb              # Validação e teste
│   ├── 04_yolo_tradicional.ipynb             # YOLO tradicional
│   ├── 05_cnn_do_zero_parte1.ipynb           # CNN do zero (Parte 1)
│   ├── 06_cnn_do_zero_parte2.ipynb           # CNN do zero (Parte 2)
│   ├── 07_cnn_do_zero_parte3.ipynb           # CNN do zero (Parte 3)
│   └── 08_comparacao_final.ipynb             # Comparação final
├── dev/                                      # Documentação de desenvolvimento
│   ├── CHECKLIST.md                          # Checklist do projeto
│   ├── dataset_guidelines.md                 # Diretrizes para datasets
│   ├── dataset_exploration.md                # Exploração de datasets
│   ├── dataset_organization.md               # Organização do dataset
│   ├── annotation_format.md                  # Formato de anotação YOLO
│   ├── training_process.md                   # Processo de treinamento
│   ├── evaluation_metrics.md                 # Métricas de avaliação
│   ├── yolo_comparison.md                    # Comparação YOLO customizado vs. tradicional
│   ├── cnn_implementation.md                 # Implementação da CNN
│   ├── final_comparison.md                   # Comparação final entre abordagens
│   ├── video_script.md                       # Roteiro para o vídeo
│   └── project_summary.md                    # Resumo do projeto
├── setup_env.sh                              # Script para configuração do ambiente
├── merge_notebooks.py                        # Script para mesclar notebooks
└── README.md                                 # Este arquivo
```

### Arquivos Principais:

1. **notebooks/01_treinamento_yolo_custom.ipynb**:
   - Configuração do ambiente para treinamento do YOLO
   - Verificação do dataset
   - Criação do arquivo de configuração YAML
   - Treinamento do modelo YOLO com 30 épocas
   - Análise dos resultados do treinamento
   - Visualização de predições

2. **notebooks/02_treinamento_yolo_custom_60epocas.ipynb**:
   - Treinamento do modelo YOLO com 60 épocas
   - Análise do impacto do aumento de épocas
   - Comparação com o modelo de 30 épocas

3. **notebooks/03_validacao_teste.ipynb**:
   - Validação dos modelos treinados
   - Testes com imagens separadas
   - Comparação de resultados entre os modelos de 30 e 60 épocas
   - Análise de métricas de desempenho

4. **notebooks/04_yolo_tradicional.ipynb**:
   - Aplicação do YOLO tradicional pré-treinado
   - Comparação com os modelos YOLO customizados
   - Análise de diferenças de desempenho

5. **notebooks/05_cnn_do_zero_parte1.ipynb**:
   - Preparação dos dados para a CNN
   - Definição da arquitetura da CNN
   - Configuração do ambiente de treinamento

6. **notebooks/06_cnn_do_zero_parte2.ipynb**:
   - Treinamento da CNN
   - Monitoramento de métricas
   - Ajuste de hiperparâmetros

7. **notebooks/07_cnn_do_zero_parte3.ipynb**:
   - Avaliação da CNN treinada
   - Comparação com abordagens YOLO
   - Análise de vantagens e limitações

8. **notebooks/08_comparacao_final.ipynb**:
   - Análise comparativa de todas as abordagens
   - Avaliação de métricas de desempenho
   - Análise de custo-benefício
   - Conclusões e recomendações para diferentes cenários

## 🗃 Histórico de lançamentos

* 1.0.0 - 05/05/2025
    * Implementação das fases 1-8 (notebooks)
    * Documentação completa do projeto
    * Vídeo demonstrativo

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/">MODELO GIT FIAP por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
