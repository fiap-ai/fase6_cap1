#!/usr/bin/env python3
"""
Script para mesclar múltiplos notebooks Jupyter em um único arquivo.
"""

import nbformat
import sys

def merge_notebooks(notebook_filenames, output_filename):
    """
    Mescla múltiplos notebooks Jupyter em um único arquivo.
    
    Args:
        notebook_filenames: Lista de caminhos para os notebooks de entrada
        output_filename: Caminho para o notebook de saída
    """
    merged = None
    
    print(f"Mesclando {len(notebook_filenames)} notebooks...")
    
    for i, filename in enumerate(notebook_filenames):
        print(f"Processando {i+1}/{len(notebook_filenames)}: {filename}")
        
        with open(filename, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
            # Se este é o primeiro notebook, usamos como base
            if merged is None:
                merged = nb
                print(f"  Usando {filename} como notebook base")
            else:
                # Para os notebooks subsequentes, adicionamos apenas as células
                # Opcionalmente, podemos adicionar uma célula de markdown como separador
                merged.cells.append(nbformat.v4.new_markdown_cell(f"## Continuação do arquivo: {filename}"))
                print(f"  Adicionando {len(nb.cells)} células de {filename}")
                merged.cells.extend(nb.cells)
    
    # Escrever o notebook mesclado
    if merged:
        with open(output_filename, 'w', encoding='utf-8') as f:
            nbformat.write(merged, f)
        print(f"\nNotebooks mesclados com sucesso em: {output_filename}")
        print(f"Total de células no notebook final: {len(merged.cells)}")
    else:
        print("Erro: Nenhum notebook foi processado.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python merge_notebooks.py output.ipynb input1.ipynb input2.ipynb ...")
        sys.exit(1)
    
    output_filename = sys.argv[1]
    notebook_filenames = sys.argv[2:]
    
    merge_notebooks(notebook_filenames, output_filename)
