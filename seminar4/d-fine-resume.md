# D-FINE: regressão de caixas por refinamento de distribuições

**Artigo:** Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun e Feng
Wu. *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution
Refinement*. ICLR 2025 Spotlight, arXiv:2410.13842 (2024).

## 1. Problema e ideia central

Detectores da família DETR produzem um conjunto de objetos por meio de *queries*
e dispensam, em sua formulação clássica, a supressão não máxima. Variantes em
tempo real, como RT-DETR, tornam essa família competitiva em latência. Ainda
assim, a regressão usual de caixas prevê coordenadas fixas, dando pouca
estrutura intermediária para ajustar separadamente cada borda quando a
localização é ambígua.

O D-FINE parte de RT-DETR e substitui o refinamento puramente pontual por duas
ideias complementares: **Fine-grained Distribution Refinement (FDR)**, que
representa correções das quatro bordas como distribuições discretas, e **Global
Optimal Localization Self-Distillation (GO-LSD)**, que usa as distribuições da
camada final para orientar camadas anteriores durante o treinamento. Ajustes
adicionais no codificador e decodificador reduzem custo. O benefício medido é
um compromisso entre AP e latência no protocolo de COCO val2017 e TensorRT FP16
em uma NVIDIA T4; esse resultado não se transfere automaticamente para outros
dispositivos ou implementações.

## 2. Contexto técnico

Em DETR, um decodificador transforma *queries* em classes e caixas. O
emparelhamento húngaro relaciona predições e anotações no treinamento. A
regressão convencional usa coordenadas de centro e dimensões $(x,y,w,h)$ ou
distâncias às bordas $(t,b,l,r)$, normalmente supervisionadas com termos L1 e
IoU. Essas perdas avaliam a caixa final, mas não descrevem uma distribuição de
correções possíveis para cada borda.

Métodos como GFocal já aplicavam distribuições discretas a distâncias de
bordas. O diferencial estudado no D-FINE é incorporá-las como **correções
residuais sucessivas** no decodificador de um detector do tipo DETR, com uma
função não uniforme para os valores dos intervalos. Assim, bins perto de zero
permitem ajustes menores, enquanto os extremos admitem correções maiores.

## 3. FDR: Fine-grained Distribution Refinement

A primeira camada prevê uma caixa inicial $b^0=(x,y,W,H)$ e distribuições para
as quatro bordas. Em cada camada seguinte, a cabeça de distribuição prevê uma
correção dos *logits* anteriores. Uma função *softmax* converte os logits
acumulados em probabilidades. Para cada borda, o deslocamento é a esperança da
distribuição sobre os valores definidos pela função de ponderação $w(n)$:

$$
p_e^{(l)}(n)=\operatorname{softmax}\bigl(z_e^{(l-1)}(n)+\Delta z_e^{(l)}(n)\bigr),
\qquad
d_e^{(l)}=d_e^{(0)}+s_e\sum_{n=0}^{N}w(n)p_e^{(l)}(n),
$$

em que $e$ identifica uma borda e $s_e$ vale a altura inicial para topo e base
ou a largura inicial para esquerda e direita. Portanto, a distribuição
representa uma **correção relativa à caixa inicial**, não uma probabilidade
direta sobre todas as caixas possíveis. A escala pela caixa inicial adapta o
deslocamento ao tamanho do objeto.

A perda **Fine-Grained Localization (FGL)** supervisiona os dois bins vizinhos
do deslocamento verdadeiro por entropia cruzada interpolada, com ponderação por
IoU. As perdas usuais de classificação e de caixa continuam presentes. Segundo
o artigo, a combinação fornece sinal de treinamento mais detalhado para cada
borda. A distribuição pode expressar ambiguidade interna, mas o artigo avalia
AP de detecção, não calibração probabilística de incerteza.

## 4. GO-LSD: auto destilação de localização

O GO-LSD usa a camada final do próprio decodificador como professora das
camadas rasas. Para selecionar as *queries* supervisionadas, reúne os índices
de emparelhamento húngaro obtidos em todas as camadas. Isso permite aproveitar
boas localizações que não seriam selecionadas se apenas o emparelhamento final
fosse considerado. A classificação preserva o emparelhamento um a um.

A perda **Decoupled Distillation Focal (DDF)** compara distribuições rasas e
finais por divergência KL com temperatura. Para predições emparelhadas, o peso
depende de IoU; para não emparelhadas, da confiança de classificação. A
normalização considera as quantidades de cada grupo. A orientação da camada
final melhora as predições iniciais e deixa correções residuais menores para
as camadas profundas. Como não há um modelo professor separado, a destilação
adiciona custo sobretudo no treinamento e não exige uma segunda rede na
inferência.

## 5. Ajustes de eficiência

Além de FDR e GO-LSD, os autores removem projeções do decodificador, inserem
uma camada de *target gating*, trocam blocos CSP por GELAN com dimensão interna
reduzida, redistribuem pontos de amostragem entre escalas e adotam a estratégia
de treinamento do RT-DETRv2. Essas alterações importam para interpretar a
comparação com RT-DETR: o D-FINE completo não é apenas RT-DETR acrescido de
duas perdas.

Na sequência de ablação da Tabela 3, o baseline RT-DETR-HGNetv2-L tem **53,0
AP, 110 GFLOPs e 9,25 ms**. Após os ajustes de eficiência, antes de FDR, o
modelo retorna a **53,0 AP, 91 GFLOPs e 7,90 ms**. FDR eleva o AP a **53,5** e
GO-LSD a **54,0**, com **91 GFLOPs e 8,07 ms**. Assim, parte da redução de custo
decorre dos ajustes arquiteturais, enquanto os ganhos finais de AP decorrem de
FDR e GO-LSD nesse experimento.

## 6. Avaliação e resultados

O AP informado é a média COCO para IoU de 0,50 a 0,95 em **COCO val2017**.
Latência é de ponta a ponta, com **TensorRT FP16, lote 1 e GPU NVIDIA T4**,
segundo o repositório oficial. FPS é aproximadamente o inverso da latência;
não equivale à velocidade em PyTorch ou CPU.

| Variante | Treinamento | AP COCO | Parâmetros | GFLOPs | Latência T4 |
| --- | --- | ---: | ---: | ---: | ---: |
| D-FINE-N | COCO | 42,8 | 4 M | 7 | 2,12 ms |
| D-FINE-S | COCO | 48,5 | 10 M | 25 | 3,49 ms |
| D-FINE-M | COCO | 52,3 | 19 M | 57 | 5,62 ms |
| D-FINE-L | COCO | 54,0 | 31 M | 91 | 8,07 ms |
| D-FINE-X | COCO | 55,8 | 62 M | 202 | 12,89 ms |
| D-FINE-L | Objects365 → COCO | 57,1 | 31 M | 91 | 8,07 ms |
| D-FINE-X | Objects365 → COCO | 59,3 | 62 M | 202 | 12,89 ms |

Os valores da família COCO vêm da tabela de checkpoints do repositório. Há
uma atualização posterior do checkpoint **D-FINE-L (E25)** que registra **57,3
AP** após Objects365 → COCO. O artigo original informa **57,1 AP** para L;
essa diferença de versão deve ser preservada ao comparar fontes.

Na Tabela 2 do artigo, acrescentar FDR e GO-LSD a outros DETRs aumenta o AP
em **2,0 a 5,3 pontos** nos cinco pares de experimentos reportados, com o maior
ganho no DAB-DETR de 12 épocas. Isso mostra transferência sob aquelas
configurações, sem demonstrar superioridade em toda arquitetura ou conjunto de
dados. Na comparação de destilação da Tabela 5, GO-LSD alcança **54,5 AP**
contra **53,0 AP** do baseline, com **31 contra 29 minutos por época** e
**8.734 contra 8.552 MB** de memória. Essa tabela usa uma configuração de
ablação sem os ajustes leves e, portanto, não deve ser confundida com os
**54,0 AP** do D-FINE-L completo.

## 7. Leitura crítica e limites

- A evidência quantitativa principal é COCO val2017; a generalização para
  domínios distintos exige novos testes.
- Os números de latência dependem de T4, TensorRT, FP16, resolução e lote.
  A implementação Transformers do notebook mede outro sistema.
- O pré treinamento em Objects365 altera o orçamento de dados. Comparações com
  modelos treinados só em COCO precisam ser separadas.
- A distribuição é uma representação interna de correção. O estudo não fornece
  uma avaliação abrangente da calibração da incerteza das caixas.
- O modelo completo combina várias mudanças arquiteturais e de treinamento;
  as ablações são essenciais para atribuir os ganhos.
- O notebook executa inferência qualitativa com checkpoint Nano e não mede AP,
  reproduz o treinamento nem implementa GO-LSD.

## 8. Síntese

O D-FINE melhora a localização em um DETR de tempo real ao refinar
distribuições discretas de deslocamento de bordas e usar auto destilação de
localização entre camadas. No protocolo do artigo, D-FINE-L e X alcançam 54,0
e 55,8 AP em COCO, com 8,07 e 12,89 ms na T4. O resultado mais importante é
que a representação e a supervisão da regressão de caixas podem elevar o AP
sem comprometer o orçamento de inferência no mesmo grau que simplesmente
aumentar o modelo.

## Fontes

1. [Artigo original e tabelas](https://arxiv.org/html/2410.13842v1).
2. [Repositório oficial, checkpoints e protocolo de latência](https://github.com/Peterande/D-FINE).
3. [Documentação da implementação em Transformers](https://huggingface.co/docs/transformers/model_doc/d_fine).
