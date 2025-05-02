Introdução
PROJETO FASE 6 – O COMEÇO DA REDE NEURAL


1) DESCRIÇÃO RÁPIDA DO PROJETO: 

Para a Fase 6, vamos botar a mão na massa no desenvolvimento de uma rede neural. Além dessa, vamos ter outras duas entregas definidas como “Ir Além”, que não valem nota.

Assim como aconteceu na fase anterior, esperamos que os grupos se desafiem em aprender ainda mais com essas duas entregas extras. Como recompensa das entregas dos “Ir Além”, os grupos ganharão gratificações (não notas) que serão explicadas ao longo das Lives e nesse documento.

 

2) DESCRIÇÃO DETALHADA DO PROJETO:

Entrega 1: a FarmTech Solutions está expandindo os serviços de IA e indo para além do agronegócio. Sua carteira de clientes aumentou e agora está prestando serviços de IA na área da saúde animal, segurança patrimonial de fazendas e de casas de seus clientes, controle de acessos de funcionários, análise de documentos de qualquer natureza. E ainda, começou a atuar na área da visão computacional.

Nessa entrega, você faz parte do time de desenvolvedores da FarmTech, e está visitando um cliente que gostaria de entender como funciona um sistema de visão computacional na prática.

Seu objetivo é criar um sistema de visão computacional usando o YOLO que demonstre seu potencial e acurácia. É importante ressaltar que você é livre para escolher o cenário de imagens que servirá para a etapa de treinamento, validação e testes.

 

Metas da entrega 1:

Organizar um dataset com no mínimo 40 imagens de um objeto A que você escolher e +40 imagens de um outro objeto B bem diferente do A, totalizando 80 imagens;
Dessas 40 imagens do objeto A, reserve 32 para a etapa do treino, 4 para validação e 4 para testes. Faça o mesmo para o banco de imagens do objeto B;
Organizar suas imagens no seu Google Drive pessoal ou do grupo, separando em pastas de treino, validação e teste;
Fazer a rotulação das imagens de treinamento usando o site Make Sense IA;
Salvar as rotulações no seu Google Drive;
Montar um Colab, conectado ao seu Google Drive, que seja capaz de fazer o treinamento, validação e teste, e que descreva em Markdown, o passo a passo dessas três etapas;
Fazer ao menos duas simulações com a quantidade diferente de épocas, e comparar os resultados de acurácia, erro e desempenho quando alteramos tais parâmetros. Escolha por exemplo, 30 e 60 épocas, isto é, bem diferentes entre si;
Apresente suas conclusões sobre os resultados encontrados na validação e nos testes. Os resultados estarão disponíveis em “Results saved to yolov5/runs/detect/expX” onde X vai incrementando a cada treino que você realizar.
Traga prints das imagens de testes processadas pelo seu modelo para convencer o seu cliente fictício da FarmTech Solutions.
 

 Entregáveis do enunciado 1:

Insira a sua solução em um novo repositório do GitHub com o nome do seu grupo (de 1 a 5 pessoas ou solo) e nos envie o link do Github via portal da FIAP. Pode usar um arquivo PDF para nos enviar o link. Pedimos que não realize nenhum novo commit após a data da entrega, para não classificar como entrega após o prazo. Além disso, nesse repositório, faça o upload do link do notebook Jupyter, pois vamos executar seu notebook na correção. O Jupyter precisa ter:
Células de códigos executadas, com o código Python otimizado e com comentários das linhas;
Células de markdown organizando seu relatório e discorrendo textualmente sobre os achados a partir dos dados, e conclusões a respeito dos pontos fortes e limitações do trabalho;
O nome do arquivo deve conter seu nome completo, RM e pbl_fase6.ipynb, por exemplo: “JoaoSantos_rm76332_pbl_fase6.ipynb”.
Vídeo: grave um vídeo de até 5 minutos demonstrando o funcionamento desse entregável, poste no YouTube de forma “não listado”, e coloque o link no seu GitHub, dentro do README.
Desenvolva o seu README com uma documentação introdutória, conduzindo o leitor para o seu Colab/Jupiter, onde lá, estará todo o passo a passo da sua solução e a sua descrição completa. Não precisa repetir a descrição do Jupiter no README do GitHub e sim, fazer uma integração documental da solução. Deixe sempre os seus repositórios públicos para que eles sejam acessíveis pela equipe interna da FIAP, mas cuidado com seus links para não vazarem e serem plagiados. 
Dica: assista esse vídeo para saber mais detalhes de como subir o Colab Notebook para o seu Git: <https://www.youtube.com/watch?v=5ZYRqca7OVc>.

 

Entrega 2: agora que você já experimentou a customização da YOLO para reconhecer objetos da sua base montada, chegou a hora de comparar o resultado com outras abordagens “concorrentes”.

Como quase tudo na computação, na Visão Computacional não existe uma solução 100% “melhor” ou “pior” que as demais: tudo é uma questão de cenários de aplicabilidade, bem como critérios de desempate, que você adquire fazendo. Assim, ainda que a YOLO tenha performado bem (ou não!) é sempre bom experimentar outras soluções.

Metas da entrega 2:

A partir da base de dados que você montou para a Entrega 1:

Aplique a YOLO tradicional, vista no capítulo 3 de Redes Neurais, e avalie a performance desta rede em relação à proposta na Entrega 1;
Treine uma CNN do zero para classificar a qual classe a imagem pertence.


Entregáveis do enunciado 2:

Para cada abordagem realizada (YOLO adaptável feito na Entrega 1, YOLO padrão e CNN treinada do zero, esses últimos disponíveis nos capítulos de Redes Neurais), avalie criticamente os resultados comparando-os em termos de:
Facilidade de uso/integração;
Precisão do modelo
Tempo de treinamento/customização da rede (se aplicável);
Tempo de inferência (predição).
Jupyter notebook ou Colab integrado ao seu Github com a implementação e avaliação da sua solução. Seu notebook deve conter:
Código executado;
Saídas;
Avaliações;
Células markdown avaliando criticamente seus resultados e comparando as soluções implementadas.
3) PROJETO “IR ALÉM”:

Nesta seção (assim como na fase anterior), vamos apresentar dois entregáveis extras (que não valem nota), onde os grupos poderão escolher qual “Ir Além” gostariam de desenvolver. Se postarem suas soluções, serão gratificados da seguinte forma: cada integrante receberá um troféu de excelência em busca do “Ir Além” no curso de IA da FIAP ao final do ano letivo, isto é, os grupos que entregarem um “Ir Além” entre as Fases 5, 6 e 7, terão seus pontos somados.

Cada “Ir Além” vale até 10 pontos, portanto, teremos 30 pontos em jogo que não impactarão no boletim, e sim, apenas um game de desafio interno na turma. A nota do “Ir Além” será divulgada particularmente no chat do Teams para o responsável que postou a solução.

No final do ano letivo, os pontos serão somados e divulgados amplamente no Teams. O que se espera de cada “Ir Além” será descrito a seguir. E ainda, as entregas que não estiverem totalmente funcionais e corretas, serão avaliadas mesmo assim de 0 a 10.

O importante é tentar ir além! A quantidade de grupos que serão gratificados ainda está em análise, pois precisamos de tempo para observar o nível do engajamento dessa proposta. Contudo, a ideia é gratificar uma boa parte dos grupos participantes.

 

3.1) Primeira opção “Ir Além” – Sistema de Coleta de Imagem usando o ESP32

Se o seu grupo reconhecer as imagens da Entrega 1 usando um ESP32-CAM real ou utilizando uma Web CAM de PC conectada ao Python e lendo o arquivo best.pt que foi gerado na Entrega 1, você e seu grupo estarão indo além. Portanto, desenvolva um projeto utilizando um ESP32-CAM real com comunicação Wi-Fi que colete imagens em tempo real e transmita para um Visual Code que reconheça tais objetos que você escolheu. Caso os objetos escolhidos não existam ao seu arredor, pode utilizar imagens de uma tela de TV para filmar com as câmeras e realizar as detecções.

 

Etapas do projeto:

Seguir com as orientações do capítulo da disciplina de AI Computer Systems & Sensors dessa fase que ensina como fazer a modelagem de imagens customizadas;
Implementar o ESP32-CAM na prática. Note que não estamos obrigando a compra do componente porque esse entregável está dentro do programa “Ir Além”. 
Critérios de avaliação do “Ir Além” (primeira opção):

Funcionalidade do sistema: coleta e envio de dados funcionais.
Integração Wi-Fi: comunicação estável e eficiente.
Escolha e justificativa do ESP32-CAM: clareza e alinhamento ao contexto.
Documentação no GitHub:
Código-fonte organizado e comentado;
Figura clara da arquitetura do projeto;
Justificativa concisa.
Apresentação final: demonstração prática do sistema funcionando por meio de um vídeo de até 5 minutos, além de apresentar seu GitHub de forma organizada.
Entregável:

Notebook Jupyter/Colab no GitHub na seção "Ir Além".
Adicione o código-fonte comentado.
Inclua uma justificativa crítica e clara desse projeto, com figura autoral explicativa, buscando a documentação completa.
Insira imagens que comprovem os resultados do processamento, da implementação, e insira comentários pertinentes.
Demonstração funcional do sistema por meio de um link de vídeo “não listado” no YouTube de até 5 minutos.
3.2) Segunda opção “Ir Além” – Usando Transfer Learning e Fine Tuning

Além de treinar uma CNN do zero numa arquitetura que você definiu, experimente também fazer o Transfer Learning de alguma grande rede treinada na ImageNet, com Fine Tuning. Nessa opção de “ir além” no seu projeto, implemente e avalie mais duas abordagens de classificação de imagens da sua base escolhida. 

Etapas do projeto:

Ao invés de treinar uma arquitetura CNN do zero, faça o Transfer Learning e Fine Tuning de alguma grande rede treinada na ImageNet. Nesta etapa, você pode se basear nas redes da família VGG, Incepction, MobileNet ou qualquer outra disponível no TensorFlow. Não deixe de investigar as implementações disponíveis na biblioteca e estudar as configurações necessárias. Sua hipótese a ser avaliada aqui é: será que uma grande rede pré-treinada em um conjunto de dados imenso performa melhor que uma arquitetura treinada do zero?
Nesta segunda abordagem, vamos facilitar o trabalho da rede de classificação, fazendo ela olhar apenas para a área da imagem que contém informações relevantes. Para isso, você deve:
Aplicar uma rede de segmentação do objeto desejado, criando uma máscara;
Usar a máscara obtida para “cortar” a imagem original, deixando apenas o objeto principal e apagando o background (por exemplo, pode manter os demais pixels como brancos ou pretos. Busque referências na Internet de como aplicar máscaras em imagens);
A partir da imagem recortada, aplique a rede de classificação desejada (aquela que você treinou no zero ou o Transfer Learning). Compare as abordagens em termos de acertos;
Sua hipótese a ser validada aqui é: será que pré-segmentar o objeto de interesse, dando às redes a informação de quais pixels devem ser utilizados na interpretação do conteúdo, facilita a classificação da imagem? 
Critérios de avaliação do “Ir Além” (segunda opção):

Implementação: implementação das duas abordagens propostas, com código limpo, organizado e comentado.
Escolhas do Transfer Learning: justificar a escolha da arquitetura de referência, realizar Fine Tuning, justificar a escolha da camada de congelamento e, por fim, justificar qualquer pré-processamento extra necessário para rodar o modelo.
Escolhas da segmentação: realizar a segmentação de forma automática, usando a rede que desejar para criar as máscaras. Demonstrar, para algumas imagens da sua base, a imagem original, a máscara obtida e a imagem com o background recortado (aplicação da máscara sobre a imagem original).
Documentação no GitHub:
Código-fonte organizado e comentado;
Avaliações textuais concisas sobre as comparações de resultados.
Apresentação final: demonstração prática do código funcionando por meio de um vídeo de até 5 minutos e apresentar seu GitHub de forma organizada. 
Entregável:

Notebook Jupyter/Colab no GitHub na seção "Ir Além".
Adicione o código-fonte comentado.
Inclua uma justificativa crítica e clara desse projeto, com figura autoral explicativa, buscando a documentação completa.
Insira imagens que comprovem os resultados do processamento, da implementação, e insira comentários pertinentes.
Demonstração funcional do sistema por meio de um link de vídeo “não listado” no YouTube de até 5 minutos. 
BAREMA DAS ENTREGAS OBRIGATÓRIAS

Critérios de avaliação das Entregas 1 e 2:

Critério

Descrição

Peso

Repositório no GitHub

O repositório foi criado no prazo, possui o notebook Jupyter/Colab correto e atende às exigências de nomeação do arquivo. Não deve haver commits após a data de entrega.

1.5

Notebook Jupyter / Colab

O notebook contém:
 - Células de código executadas, com código Python otimizado e funcional.
 - Células de markdown organizadas, explicando os achados, os pontos fortes e as limitações do trabalho.

3.0

Estrutura do README

O README possui:
 - Documentação introdutória clara.
 - Link funcional para o notebook Jupyter.
 - Link para o vídeo demonstrativo no YouTube.

2.0

Vídeo Demonstrativo

O vídeo atende às exigências:
 - Duração de até 5 minutos.
 - Demonstra o funcionamento do entregável de forma clara.
 - Postado no YouTube como “não listado” e incluído no README.

2.0

Organização e Apresentação Geral

Clareza e organização geral do projeto:
 - Estrutura do repositório.
 - Nome correto dos arquivos.
 - Notebook bem-organizado e de fácil leitura.

1.5

 

Detalhes adicionais:

Notebook Jupyter:
Código Python deve ser otimizado, funcional e executado corretamente.
As células markdown devem conter explicações textuais bem estruturadas sobre resultados, achados, conclusões e limitações.
README:
Deve guiar o leitor para o notebook Jupyter.
Não repetir conteúdos já presentes no Jupyter.
Vídeo:
Deve ser breve, claro e focado no funcionamento da solução entregue.
O link deve estar no README.
Pontualidade:
Entrega no prazo sem commits adicionais é um requisito para nota máxima.
Avaliação:
9 a 10 pontos: todos os critérios atendidos com excelência, código funcional, organização excelente, comparações bem concluídas e vídeo demonstrativo claro.
7 a 8,9 pontos: entrega completa, mas com pequenas falhas de organização ou execução.
5 a 6,9 pontos: projeto entregue com falhas significativas (por exemplo, código não funcional ou documentação incompleta).
0 a 4,9 pontos: entrega não atende aos requisitos mínimos ou está fora do prazo.