# RF-DETR: busca de arquitetura neural para Detection Transformers em tempo real

## Resumo

Detectores de vocabulário aberto alcançam desempenho impressionante no COCO,
mas frequentemente não generalizam para conjuntos de dados reais com classes
fora da distribuição, ausentes de seu pré-treinamento. Em vez de simplesmente
ajustar um modelo visão-linguagem (VLM) pesado para novos domínios, apresentamos
o RF-DETR, um Detection Transformer especialista e leve que descobre curvas de
Pareto entre acurácia e latência para qualquer conjunto-alvo por meio de busca
de arquitetura neural (NAS) com compartilhamento de pesos.

Nossa abordagem ajusta uma rede-base pré-treinada ao conjunto-alvo e avalia
milhares de configurações com diferentes relações entre acurácia e latência sem
retreiná-las. Também revisitamos os “controles ajustáveis” da NAS para melhorar
a transferência de DETRs a domínios diversos. O RF-DETR melhora
significativamente o estado da arte em métodos de tempo real no COCO e no
Roboflow100-VL. O RF-DETR Nano alcança 48,0 AP no COCO, 5,3 AP acima do D-FINE
Nano com latência semelhante. O RF-DETR 2x-Large supera o GroundingDINO Tiny em
1,2 AP no Roboflow100-VL e é vinte vezes mais rápido. Até onde sabemos, é o
primeiro detector em tempo real a ultrapassar 60 AP no COCO.

## 1. Introdução

A detecção de objetos é um problema fundamental de visão computacional que
amadureceu nos últimos anos. Detectores de vocabulário aberto, como
GroundingDINO e YOLO-World, apresentam excelente desempenho *zero-shot* em
categorias comuns, mas VLMs ainda têm dificuldade em classes, tarefas e
modalidades de imagem fora da distribuição de pré-treinamento. Ajustar esses
modelos aumenta o desempenho no domínio, porém custa eficiência por causa dos
codificadores de texto pesados e reduz a generalização de vocabulário aberto.
Detectores especialistas, como D-FINE e RT-DETR, são rápidos, mas ficam atrás de
VLMs ajustados. Modernizamos os detectores especialistas combinando
pré-treinamento em escala de internet com arquiteturas de tempo real.

**Detectores especialistas estão superotimizados para o COCO?** Benchmarks como
PASCAL VOC e COCO sustentaram o progresso da área. Contudo, detectores recentes
acabaram implicitamente sobreajustados ao COCO por meio de arquiteturas,
agendadores de taxa de aprendizado e aumentos de dados específicos. Modelos
como YOLOv8 generalizam mal para conjuntos reais cujas distribuições diferem em
número de objetos, classes e imagens. Para enfrentar isso, o RF-DETR elimina
agendadores e usa pré-treinamento em escala de internet, além de NAS para se
especializar em diferentes plataformas e conjuntos.

**Repensando NAS para DETRs.** NAS descobre relações entre acurácia e latência
dentro de um espaço de busca. Trabalhos anteriores estudaram classificação ou
subcomponentes de detectores. Aqui, exploramos NAS de ponta a ponta com
compartilhamento de pesos para detecção e segmentação. Inspirados em OFA,
variamos tanto entradas, como resolução, quanto componentes, como tamanho do
*patch*, durante o treinamento. Na inferência, variamos camadas do decodificador
e *query tokens*. A busca ocorre somente depois que o modelo-base foi treinado
no conjunto-alvo; todas as sub-redes mantêm bom desempenho sem ajuste adicional.
Até configurações não vistas durante o treinamento funcionam bem. Para
segmentação, adicionamos apenas uma cabeça leve, formando o RF-DETR-Seg.

**Padronização da latência.** No COCO e no RF100-VL, RF-DETR obtém resultados de
estado da arte entre detectores em tempo real. Comparar latências publicadas,
porém, é difícil porque cada trabalho mede de maneira diferente. Identificamos
o controle térmico e de potência da GPU como fonte central de falta de
reprodutibilidade. Inserir um intervalo entre passagens reduz o consumo
excessivo e estabiliza a medição.

As três principais contribuições são:

1. Uma família de detectores e segmentadores sem agendamento, baseada em NAS,
   que supera métodos anteriores no RF100-VL e no COCO até 40 ms de latência.
2. Uma análise dos controles ajustáveis da NAS com compartilhamento de pesos
   para melhorar a relação acurácia-latência em detecção de ponta a ponta.
3. Uma revisão dos protocolos de latência e um procedimento simples para tornar
   os resultados mais reprodutíveis.

## 2. Trabalhos relacionados

**Busca de arquitetura neural.** NAS identifica automaticamente famílias de
arquiteturas com diferentes relações entre acurácia e latência. Métodos antigos
maximizavam sobretudo a acurácia e produziam redes caras. Métodos recentes,
conscientes do hardware, incluem a plataforma no processo, mas repetem busca e
treinamento a cada novo dispositivo. OFA separa treinamento e busca ao otimizar
milhares de sub-redes com pesos compartilhados. Em detecção, a NAS costuma
apenas substituir o backbone; nosso método otimiza diretamente a detecção de
ponta a ponta e encontra configurações de Pareto para qualquer conjunto-alvo.

**Detectores em tempo real.** Detectores de dois estágios historicamente
priorizaram acurácia, enquanto os de um estágio, como YOLO e SSD, priorizaram
velocidade. Variantes modernas de YOLO melhoram arquitetura, aumento de dados e
treinamento, mas em geral dependem de supressão não máxima (NMS). DETR elimina
NMS e âncoras manuais, porém suas primeiras versões eram lentas. RT-DETR e
LW-DETR adaptaram a família para tempo real. Com base no LW-DETR, o RF-DETR é o
primeiro detector em tempo real acima de 60 AP no COCO.

**Modelos visão-linguagem.** VLMs aprendem com pares imagem-texto em escala de
internet, o que viabiliza detecção de vocabulário aberto. GLIP, Detic e MQ-Det
são exemplos. Esses modelos têm bom desempenho *zero-shot*, mas falham em
categorias ausentes do pré-treinamento e são caros. O RF-DETR combina a
inferência rápida dos especialistas com os conhecimentos prévios dos VLMs.

## 3. RF-DETR: NAS com pesos compartilhados e modelos fundacionais

### 3.1. Arquitetura-base

O RF-DETR moderniza o LW-DETR simplificando arquitetura e treinamento para
generalizar melhor. Substituímos o backbone CAEv2 pelo DINOv2. Inicializar com
pesos do DINOv2 melhora muito a detecção em conjuntos pequenos. Seu codificador
tem 12 camadas, contra 10 no CAEv2, e é mais lento; recuperamos essa diferença
com NAS. Para permitir acumulação de gradientes em GPUs de consumo, usamos
*layer normalization* em vez de *batch normalization* no projetor multiescala.

**Segmentação de instâncias em tempo real.** Adicionamos uma cabeça leve que
interpola bilinearmente a saída do codificador e aprende um projetor para criar
um mapa de *embeddings* de pixels. A mesma característica de baixa resolução é
ampliada para detecção e segmentação, preservando informação espacial útil. Para
reduzir latência, a cabeça não usa características multiescala do backbone. O
produto interno dos *embeddings* de cada *query token*, transformados por uma
FFN na saída de cada camada do decodificador, com o mapa de pixels produz as
máscaras. Esses *embeddings* podem ser interpretados como protótipos de
segmentação. O RF-DETR-Seg é pré-treinado no Objects-365 pseudo-rotulado pelo
SAM2.

**Figura 2 — Arquitetura RF-DETR.** Um backbone ViT pré-treinado extrai
características multiescala. Blocos de atenção com e sem janelas são
intercalados para equilibrar acurácia e latência. A atenção cruzada deformável e
a cabeça de segmentação interpolam a saída do projetor. Perdas de detecção e
segmentação são aplicadas em todas as camadas do decodificador, permitindo
removê-las na inferência.

### 3.2. Busca de arquitetura neural de ponta a ponta

A NAS avalia milhares de configurações que variam resolução, tamanho de patch,
janelas de atenção, camadas do decodificador e *query tokens*. A cada iteração,
uma configuração aleatória é amostrada uniformemente e recebe uma atualização
de gradiente. Isso treina milhares de sub-redes em paralelo, de modo semelhante
a um conjunto com *dropout*, e age como regularização por “aumento de
arquitetura”. Segundo os autores, esta é a primeira NAS de ponta a ponta com
pesos compartilhados aplicada à detecção e segmentação.

Os controles são:

- **Tamanho do patch:** patches menores aumentam a acurácia e o custo. Uma
  transformação no estilo FlexiViT interpola tamanhos durante o treinamento.
- **Número de camadas do decodificador:** há perda de regressão em todas as
  camadas, permitindo remover qualquer quantidade na inferência. Remover todas
  transforma o RF-DETR, na prática, em um detector de estágio único e também
  reduz a cabeça de segmentação.
- **Número de query tokens:** queries aprendem conhecimentos espaciais prévios.
  Na inferência, descartamos tokens ordenados pela maior sigmoide do logit de
  classe correspondente na saída do codificador. O número ótimo reflete a
  quantidade média de objetos por imagem.
- **Resolução:** alta resolução favorece objetos pequenos; baixa resolução
  reduz o tempo. Embeddings posicionais são pré-alocados para o maior caso e
  interpolados nos demais.
- **Janelas por bloco:** a atenção em janela restringe a autoatenção a tokens
  vizinhos. Adicionar ou remover janelas controla mistura global, acurácia e
  custo.

Na inferência, selecionamos uma configuração e, portanto, um ponto da curva de
Pareto. Contagens de parâmetros semelhantes podem esconder latências muito
diferentes. Ajustar novamente os modelos minerados traz pouco ganho no COCO e
ganhos modestos no RF100-VL. Esse ajuste é opcional. A provável razão do ganho
no RF100-VL é que a forte regularização exige mais de cem épocas para convergir
em conjuntos pequenos.

### 3.3. Agendadores e aumentos enviesam o desempenho

Detectores de ponta exigem ajuste cuidadoso para benchmarks padrão, criando
viés para propriedades específicas como o número de imagens. Agendamentos
cosseno presumem um horizonte de otimização conhecido, algo impraticável nos
cem conjuntos diversos do RF100-VL. Aumentos de dados também incorporam
suposições. Inversão vertical, por exemplo, pode prejudicar domínios críticos:
um detector de pessoas em veículo autônomo não deveria aprender com imagens
invertidas e confundir reflexos em poças. Limitamos os aumentos a inversões
horizontais e recortes aleatórios.

O redimensionamento por imagem do LW-DETR preenche cada amostra até a maior do
lote, desperdiçando computação e introduzindo artefatos de janela. Redimensionar
no nível do lote reduz pixels de preenchimento e garante que todas as resoluções
de codificação posicional sejam igualmente prováveis no treinamento.

## 4. Experimentos

### 4.1. Configuração

Avaliamos no COCO, para comparação com trabalhos anteriores, e no RF100-VL,
para medir generalização a cem distribuições reais. Usamos `pycocotools` para
mAP, AP50, AP75 e AP de objetos pequenos, médios e grandes. Eficiência é medida
em GFLOPs, parâmetros e latência numa NVIDIA T4 com TensorRT 10.4 e CUDA 12.4.
Modelos de latência semelhante pertencem à mesma categoria de tamanho,
independentemente do número de parâmetros.

### 4.2. Padronização do benchmark de latência

Comparações anteriores são inconsistentes. Modelos YOLO frequentemente omitem
NMS da latência; modelos de segmentação YOLO medem somente os protótipos, não as
máscaras utilizáveis. A latência do LW-DETR reportada pelo D-FINE é 25% menor
que a original. Observamos que aquecimento e limitação de potência da GPU
explicam boa parte da diferença. Uma pausa de 200 ms entre passagens estabiliza
a medição, embora não represente vazão sustentada.

Além disso, trabalhos medem latência em FP16 e acurácia em FP32. Quantização
ingênua pode derrubar o desempenho para perto de zero AP. Defendemos medir
acurácia e latência com o mesmo artefato. A Tabela 1 mostra, por exemplo, que
RF-DETR-M em FP16 obtém 54,7 AP e 4,4 ms; D-FINE-M, 55,0 AP e 5,4 ms; e
LW-DETR-M, 52,6 AP e 4,4 ms.

### 4.3. COCO

Na detecção, RF-DETR Nano alcança 48,0 AP a 2,3 ms, mais de 5 AP acima de
D-FINE Nano e LW-DETR Tiny. O Small alcança 52,9 AP a 3,5 ms; o Medium, 54,7
AP a 4,4 ms; e o 2XL, 60,1 AP a 17,2 ms. O Nano iguala aproximadamente os
YOLOv8/YOLOv11 Medium, embora pertença à menor faixa de latência.

Na segmentação de instâncias, RF-DETR-Seg Nano obtém 40,3 AP a 3,4 ms e supera
todas as variantes YOLOv8/YOLOv11 apresentadas. Supera FastInst em 5,4 AP e é
quase dez vezes mais rápido. RF-DETR-Seg Medium chega a 45,3 AP a 5,9 ms,
aproximando-se do MaskDINO, que leva 242 ms. A variante 2XL alcança 49,9 AP a
21,8 ms.

### 4.4. RF100-VL

O RF100-VL reúne cem conjuntos de detecção. O RF-DETR 2XL obtém 63,3 AP a 15,6
ms, ou 63,5 AP após ajuste, superando GroundingDINO Tiny e LLMDet Tiny, ambos
com 62,3 AP e cerca de 309 ms. RT-DETR supera D-FINE em AP50, indício de que os
hiperparâmetros do D-FINE podem estar sobreajustados ao COCO. YOLOv8 e YOLOv11
ficam consistentemente atrás dos DETRs e não melhoram ao aumentar de escala.

### 4.5. Impacto da NAS

Hiperparâmetros mais suaves que os do LW-DETR — taxa menor e troca de batch
norm por layer norm — reduzem o AP de 52,6 para 51,6. A troca do CAEv2 pelo
DINOv2 eleva a 53,6; pré-treinamento adicional no Objects-365 leva a 54,3; NAS
com pesos compartilhados, a 54,6. Assim, o modelo final ganha cerca de 2 AP
contra o LW-DETR sem aumentar a latência. Curiosamente, a NAS também melhora a
configuração-base mesmo quando seu patch 14 não pertence ao espaço de treino.

### 4.6. Backbone, práticas de avaliação e limitações

DINOv2 apresenta o melhor backbone, superando CAEv2 em 2,4 AP. Apesar de menos
parâmetros que SigLIPv2, Hiera-S do SAM2 é mais lento, em contraste com sua
vantagem reportada. A explicação proposta é que Hiera não considerou kernels de
Flash Attention altamente otimizados no TensorRT. Também faltam variantes ViT-S
e ViT-T em muitas famílias fundacionais, dificultando uso em tempo real.

Usar a validação do COCO simultaneamente para seleção e avaliação favorece
sobreajuste. D-FINE faz busca extensa nesse conjunto e supera RT-DETR ali, mas
fica atrás dele no teste RF100-VL. Como o RF-DETR lidera entre modelos de tempo
real nos dois benchmarks, os autores recomendam conjuntos com divisões públicas
de validação e teste.

Mesmo controlando temperatura e potência, a compilação não determinística do
TensorRT gera variação de até 0,1 ms. Recompilar o mesmo ONNX pode produzir
engines ligeiramente diferentes; por isso, latências são reportadas com uma
única casa decimal.

## 5. Conclusão

Apresentamos RF-DETR, método baseado em NAS para ajustar detectores especialistas
de ponta a ponta a conjuntos e plataformas diversos. Ele supera métodos de
tempo real no COCO e no RF100-VL, incluindo ganho de aproximadamente 5 AP sobre
D-FINE Nano no COCO. Arquiteturas e agendadores atuais são moldados para o
COCO; a comunidade deveria avaliar em conjuntos grandes e diversos para evitar
sobreajuste implícito. Por fim, destacamos a alta variância dos benchmarks de
latência causada por limitação de potência e propomos um protocolo padronizado.

# Apêndices

## A. Detalhes de implementação

### Hiperparâmetros de treinamento

O RF-DETR estende LW-DETR para NAS. Objects-365 é pseudo-rotulado com SAM2 para
pré-treinar as cabeças de detecção e segmentação nos mesmos dados. A taxa de
aprendizado é 1e-4, contra 4e-4 no LW-DETR, e o lote é 128. Usa-se agendamento
EMA, sem aquecimento da taxa. Gradientes acima de 0,1 são limitados e aplica-se
decaimento multiplicativo de 0,8 por camada para preservar sobretudo as camadas
iniciais do DINOv2.

Blocos de atenção em janela ficam entre as camadas {0, 1, 3, 4, 6, 7, 9, 10};
no LW-DETR, entre {0, 1, 3, 6, 7, 9}. Blocos contíguos evitam uma operação extra
de remodelagem. O intervalo multiescala vai de 0,5 a 1,5, simétrico ao redor da
escala padrão, contra 0,7 a 1,4 no LW-DETR. Aqui, resolução é um controle da NAS,
não apenas aumento de dados.

### Avaliação de latência

Acurácia e latência são medidas com o mesmo artefato. Grafos CUDA no TensorRT
pré-enfileiram kernels em vez de lançá-los sequencialmente pela CPU. RT-DETR,
LW-DETR e RF-DETR se beneficiam; D-FINE, não. Com essa otimização, LW-DETR e
D-FINE ficam na mesma curva acurácia-latência.

### Configurações de Pareto no COCO

| Tamanho | Resolução | Patch | Janelas | Camadas do decoder | Queries | Backbone |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| N | 384 | 16 | 2 | 2 | 300 | DINOv2-S |
| S | 512 | 16 | 2 | 3 | 300 | DINOv2-S |
| M | 576 | 16 | 2 | 4 | 300 | DINOv2-S |
| L | 704 | 16 | 2 | 4 | 300 | DINOv2-S |
| XL | 700 | 20 | 1 | 5 | 300 | DINOv2-B |
| 2XL | 880 | 20 | 2 | 5 | 300 | DINOv2-B |
| Max | 828 | 12 | 1 | 6 | 300 | DINOv2-B |

Para RF-DETR-Seg, as configurações N/S/M/L/XL/2XL/Max usam resoluções
312/384/432/504/624/768/890, patches 12/12/12/12/12/12/10, janelas
1/2/2/2/2/2/1, camadas 4/4/5/5/6/6/6 e queries 100/100/200/300/300/300/300,
sempre com DINOv2-S.

No treinamento, amostram-se 11 resoluções (320 a 960), 7 patches (8, 10, 12,
16, 20, 24 e 32), 6 camadas, 1/2/4 janelas e 300 queries. Na inferência,
adiciona-se patch 14, usam-se de 0 a 6 camadas e 50/100/200/300 queries. A busca
avalia 6.468 configurações. O treinamento leva de duas a quatro vezes o de uma
base sem NAS, mas gera todos os tamanhos em uma única execução. A estimativa da
busca é 10 mil horas de GPU: 200 T4 por 48 horas.

## B. Ablação de queries e camadas do decodificador

O RF-DETR Nano é treinado com 300 queries, embora muitos conjuntos tenham menos
objetos. Em vez de escolher previamente outro número, descartamos na inferência
as queries de menor confiança na saída do codificador. Remover as cem menos
confiáveis quase não reduz desempenho e melhora modestamente a latência em
todas as profundidades. Como cada camada é supervisionada separadamente, também
podemos podá-las no teste. Remover todas elimina atenção cruzada e autoatenção
entre queries, aproximando o modelo de um YOLO de estágio único sem NMS.
Eliminar a última camada reduz a latência em 10% com queda de apenas 2 mAP.

## C. Benchmark de FLOPs

Usamos `FlopCounterMode` do PyTorch para RF-DETR, GroundingDINO e LLMDet. Ele
reproduz bem as ferramentas específicas de YOLOv11, D-FINE e LW-DETR e é mais
confiável que CalFLOPs. A contagem obtida para LW-DETR é quase o dobro da
publicada, provavelmente porque o artigo original reporta MACs como FLOPs. Para
D-FINE-S, os valores publicado/CalFLOPs/FlopCounterMode são
25,2/25,2/25,5 M; para LW-DETR-S, 16,6/22,9/31,8 M; para YOLO11-S,
21,5/23,9/21,6 M.

## D. Impacto dos nomes de classes em detectores de vocabulário aberto

Seria intuitivo que GroundingDINO aproveitasse melhor o pré-treinamento ao
receber “carro, caminhão, ônibus” em vez de índices “0, 1, 2”. Os nomes fornecem
ao VLM informação que detectores comuns não recebem. Porém, no RF100-VL, o
ajuste com nomes padrão alcança 62,3 AP e com índices, 62,5 AP. O ajuste ingênuo
de ponta a ponta parece apagar a vantagem do pré-treinamento aberto. Trabalhos
futuros devem preservar melhor esse conhecimento.

## E. Variantes maiores

LW-DETR e D-FINE projetam manualmente variantes maiores; RF-DETR as descobre na
grade da NAS. Comparamos famílias DINOv2-S e DINOv2-B a variantes D-FINE de
latência semelhante. DINOv2-S começa à frente nos modelos pequenos, mas perde a
vantagem em escalas maiores. DINOv2-B apresenta a tendência inversa: a diferença
diminui com a latência e o 2XL supera D-FINE em 0,8 AP.

No COCO, RF-DETR L/XL/2XL/Max obtêm 56,5/58,6/60,1/61,8 AP com
6,8/11,5/17,2/98,0 ms. Na segmentação, obtêm 47,1/48,8/49,9/50,5 AP com
8,8/13,5/21,8/95,6 ms. No RF100-VL, a variante XL alcança 62,6 AP, ou 63,0
após ajuste, e a 2XL alcança 63,3 ou 63,5. Novas variantes de maior latência
podem ser amostradas da mesma busca sem retreinamento.

## F. Sensibilidade de cada controle

Variar resolução e patch produz fronteiras de Pareto claras, coerentes com
FlexiViT. Configurações vistas e não vistas seguem a mesma tendência. O RF-DETR
interpola suavemente para resoluções e patches inéditos, generalizando além das
arquiteturas encontradas no treinamento.

## G. Impacto do ajuste após NAS no COCO

O ajuste posterior traz pouco benefício no COCO. O “aumento de arquitetura” da
NAS é uma regularização forte; removê-la durante o ajuste favorece sobreajuste.
Em detecção, os ganhos de AP para N/S/M/L/XL/2XL são apenas
+0,4/+0,1/+0,0/+0,0/+0,3/+0,1. Em segmentação, apenas Nano melhora 0,1 AP; as
demais variantes não melhoram. RF100-VL se beneficia mais, provavelmente por
precisar de mais de cem épocas. Reduzir configurações ou treinar NAS por mais
tempo pode ajudar.

## H. Características do conjunto e controles ajustáveis

No RF100-VL, analisamos tamanho de objetos, locais espaciais, camadas, janelas,
classes, anotações, objetos por imagem e queries. Há correlações entre número de
classes e camadas, objetos por imagem e queries, e locais espaciais e janelas.
A mais forte é entre locais espaciais e janelas (R² 0,492). Tamanho de objetos,
número de anotações e objetos por imagem quase não explicam a profundidade do
decodificador. As relações são dispersas: características do conjunto dão
intuição, mas nenhuma determina sozinha a configuração ótima.

## I. Arquitetura fixa no RF100-VL

Transferimos ao RF100-VL arquiteturas otimizadas no COCO. Mesmo sem NAS
específica, elas superam LW-DETR; a variante Large fixa atinge 62,2 AP. Contudo,
a busca no conjunto-alvo ainda traz ganhos importantes, especialmente nos
tamanhos Nano, Small e Medium, e o ajuste adicional melhora consistentemente.
A diferença entre LW-DETR e arquitetura fixa é comparável à diferença entre a
fixa e a otimizada para o alvo.

## J. Ablação de backbone no RF20-VL

Repetimos a análise do backbone em vinte conjuntos. CAEv2 obtém 64,4 AP;
DINOv2-S, 65,2; SigLIPv2, 62,2; e Hiera-S do SAM2, 65,2. As tendências do artigo
principal se mantêm, com DINOv2 oferecendo melhor equilíbrio de desempenho e
latência.

## K. Análise do intervalo entre inferências

Intervalos acima de 200 ms não alteram a latência. Para YOLOv8-M, YOLOv11-M,
RT-DETR-R18, LW-DETR-M, D-FINE-M e RF-DETR-M, os valores com 200 ms são,
respectivamente, 5,4, 5,0, 4,4, 4,3, 5,4 e 4,4 ms. A pausa aumenta muito o tempo
total e não mede vazão; trabalhos futuros devem buscar outra forma de controlar
a limitação de potência.

## L. Arquiteturas descobertas de destaque

Todos os controles aparecem em famílias de Pareto, validando o espaço de busca.
O patch tende a ser constante por família: 16 no RF-DETR com DINOv2-S, 20 com
DINOv2-B e 12 no RF-DETR-Seg com DINOv2-S. Os modelos escalam conjuntamente o
codificador — patch, janelas e resolução — e o decodificador — profundidade e
queries. No COCO, RF-DETR aumenta a profundidade e mantém queries; RF-DETR-Seg
aumenta ambos, formando um decodificador estreito e profundo.

O desempenho se correlaciona mais com o total de posições espaciais (resolução
dividida pelo patch) do que com cada variável isolada. Uma família alternativa
fixou resolução 640 e usou patches 27, 21 e 18 em Nano, Small e Medium; os
resultados ficaram quase idênticos aos de Pareto, mesmo 27 e 18 não tendo sido
vistos. Isso não vale igualmente para segmentação, pois suas características são
sempre ampliadas para um quarto da resolução de entrada.

A maioria dos RF-DETRs ótimos usa duas janelas, enquanto LW-DETR usa quatro. O
CAEv2 do LW-DETR não possui token de classe; DINOv2 depende dele. No RF-DETR, o
token é duplicado para cada janela e os tokens de janela interagem na atenção
global. Mais janelas duplicam mais tokens e reduzem a eficiência. Por fim,
modelos de baixa latência do RF100-VL usam menos queries que equivalentes do
COCO, coerente com seu menor número de objetos por imagem.

## M. Visualização das predições

Na Figura 8, RF-DETR Nano produz menos falsos positivos que LW-DETR Tiny — por
exemplo, evita confundir uma placa com uma pessoa. RF-DETR-Seg Nano também
prediz contornos mais precisos que YOLOv11 Nano.