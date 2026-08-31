# RepViT: revisitando CNNs móveis sob a perspectiva dos ViTs

> Tradução para português brasileiro do artigo *RepViT: Revisiting Mobile CNN
> From ViT Perspective* (arXiv:2307.09283v8). Os números das tabelas, as
> equações, os nomes próprios de modelos e as referências bibliográficas foram
> preservados. Consulte o [PDF original](2307.09283v8.pdf) para as figuras e a
> diagramação oficiais.

**Autores:** Ao Wang, Hui Chen, Zijia Lin, Jungong Han e Guiguang Ding.

## Resumo

Recentemente, Vision Transformers (ViTs) leves demonstraram desempenho superior
e menor latência, em comparação com redes neurais convolucionais (CNNs) leves,
em dispositivos móveis com recursos limitados. Pesquisadores descobriram muitas
conexões estruturais entre ViTs leves e CNNs leves. Contudo, as diferenças
arquiteturais relevantes entre eles — na estrutura dos blocos e nos projetos
macro e micro — ainda não haviam sido examinadas adequadamente.

Neste estudo, revisitamos o projeto eficiente de CNNs leves sob a perspectiva
dos ViTs e enfatizamos seu potencial para dispositivos móveis. Especificamente,
melhoramos de forma incremental a adequação móvel de uma CNN leve padrão, a
MobileNetV3, incorporando projetos arquiteturais eficientes dos ViTs leves. O
resultado é uma nova família de CNNs puras e leves, denominada RepViT.

Experimentos abrangentes mostram que a RepViT supera os ViTs leves existentes
no estado da arte e apresenta latência favorável em diversas tarefas de visão.
No ImageNet, a RepViT alcança mais de 80% de acurácia top-1 com latência de 1,0
ms em um iPhone 12 — a primeira vez que um modelo leve atinge esse resultado,
até onde sabemos. Além disso, ao combinar RepViT com SAM, o RepViT-SAM alcança
inferência quase dez vezes mais rápida que o avançado MobileSAM. Código e
modelos estão disponíveis em <https://github.com/THU-MIG/RepViT>.

## 1. Introdução

Em visão computacional, projetar modelos leves é uma preocupação central para
obter desempenho elevado com custo computacional reduzido. Isso é
particularmente importante em dispositivos móveis com recursos limitados, nos
quais se deseja implantar modelos de visão na borda. Na última década, a
pesquisa concentrou-se principalmente em CNNs leves e obteve avanços
significativos. Foram propostos princípios como convoluções separáveis [27],
gargalos residuais invertidos [53], embaralhamento de canais [44, 75] e
reparametrização estrutural [13, 14]. Esses princípios levaram a modelos
representativos como MobileNets, ShuffleNets e RepVGG.

Nos últimos anos, Vision Transformers surgiram como alternativa promissora às
CNNs para aprender representações visuais. Eles demonstraram desempenho
superior em classificação, segmentação semântica e detecção de objetos.
Entretanto, a tendência de aumentar o número de parâmetros para melhorar o
desempenho produz modelos grandes e de alta latência, inadequados para
dispositivos móveis. Reduzir diretamente um ViT até o tamanho permitido por
esses dispositivos costuma torná-lo inferior a uma CNN leve. Por isso, passou-se
a investigar como construir ViTs leves capazes de superar essas CNNs.

Diversos princípios foram propostos para elevar a eficiência dos ViTs em
dispositivos móveis. Algumas abordagens combinam camadas convolucionais com ViTs
em redes híbridas; outras introduzem autoatenção de complexidade linear e
projetos com dimensões consistentes. Esses trabalhos mostram que ViTs leves
podem obter latência menor e desempenho maior que CNNs leves.

Apesar desse sucesso, os ViTs leves ainda enfrentam dificuldades práticas por
causa do suporte insuficiente de hardware e bibliotecas computacionais. Também
são sensíveis a entradas de alta resolução, o que aumenta a latência. CNNs, por
outro lado, usam convoluções altamente otimizadas e de complexidade linear em
relação à entrada, vantagem importante na borda. Assim, torna-se necessário
comparar cuidadosamente CNNs e ViTs leves para projetar CNNs de alto desempenho.

As duas famílias possuem semelhanças estruturais. Ambas usam módulos
convolucionais para aprender representações espacialmente locais. Para
representações globais, CNNs leves geralmente ampliam o kernel, enquanto ViTs
leves usam autoatenção multi-cabeças (MHSA). Ainda assim, permanecem diferenças
marcantes na estrutura dos blocos e nos elementos macro e micro, pouco
investigadas até então. ViTs leves costumam usar a estrutura MetaFormer, ao
passo que CNNs leves favorecem gargalos residuais invertidos.

Isso leva à pergunta: **os projetos arquiteturais de ViTs leves podem melhorar o
desempenho das CNNs leves?** Para respondê-la, revisitamos o projeto de CNNs
leves incorporando elementos arquiteturais dos ViTs. O objetivo é aproximar as
duas famílias e destacar a adequação das CNNs à implantação móvel.

Partimos da MobileNetV3-L e modernizamos gradualmente sua arquitetura. O
resultado é a RepViT, composta inteiramente por convoluções reparametrizáveis em
uma estrutura MetaFormer semelhante à de um ViT. A RepViT apresenta desempenho
e eficiência superiores aos ViTs leves existentes em classificação no
ImageNet, detecção e segmentação de instâncias no COCO-2017 e segmentação
semântica no ADE20K. A RepViT-M2.3 alcança 83,7% de acurácia com apenas 2,3 ms
de latência; combinada ao SAM, oferece inferência quase dez vezes mais rápida
que o MobileSAM e melhor transferência *zero-shot*.

## 2. Trabalhos relacionados

Na última década, CNNs tornaram-se a abordagem predominante em visão
computacional graças a seus vieses indutivos de localidade e equivariância a
translações. Como convoluções convencionais têm custo elevado, foram propostas
técnicas como convoluções separáveis, gargalos invertidos, embaralhamento de
canais e reparametrização estrutural. Elas deram origem a CNNs leves amplamente
utilizadas, entre elas MobileNet, ShuffleNet e RepVGG.

O Vision Transformer adaptou a arquitetura Transformer para reconhecimento
visual e superou CNNs em tarefas de grande escala. Trabalhos posteriores
incorporaram vieses espaciais, operações de atenção mais eficientes e suporte a
diversas tarefas de visão. A maioria dos ViTs, porém, continuou pesada em
computação e memória. Isso motivou modelos como MobileViT, que combina blocos
MobileNet e MHSA, e EfficientFormer, que usa dimensões consistentes para
melhorar a fronteira entre latência e desempenho.

O sucesso dos ViTs leves costuma ser atribuído à capacidade da MHSA de aprender
representações globais. No entanto, diferenças arquiteturais de bloco, macro e
micro entre ViTs e CNNs também importam e costumam ser ignoradas. Este trabalho
se distingue por integrar deliberadamente os projetos arquiteturais dos ViTs a
uma CNN leve, em vez de simplesmente introduzir atenção.

## 3. Metodologia

Partimos da MobileNetV3-L e a modernizamos em diferentes granularidades. Primeiro
definimos uma medida de latência móvel e alinhamos a receita de treinamento à
dos ViTs leves. Depois exploramos o projeto de bloco, os elementos macro —
*stem*, camadas de redução de resolução, classificador e proporção entre
estágios — e, por fim, os elementos micro — tamanho de kernel e posicionamento
das camadas *squeeze-and-excitation* (SE). Todos os modelos são treinados e
avaliados no ImageNet-1K.

### 3.1. Preliminares

**Métrica de latência.** FLOPs e tamanho do modelo não se correlacionam bem com
a latência real em aplicações móveis. Por isso, medimos diretamente a latência
no dispositivo: um iPhone 12, usando Core ML Tools como compilador. Para evitar
operações sem suporte, usamos ativação GeLU na MobileNetV3-L. Sua latência
inicial é 1,01 ms.

**Alinhamento da receita de treinamento.** ViTs leves recentes costumam seguir
a receita do DeiT: AdamW, agendamento cosseno, 300 épocas, professor
RegNetY-16GF para destilação, Mixup, autoaugmentação, apagamento aleatório e
*label smoothing*. Para uma comparação justa, alinhamos a receita da
MobileNetV3-L, inicialmente sem destilação. Ela obtém 71,5% de acurácia top-1.

### 3.2. Projeto do bloco

**Separação do misturador de tokens e do misturador de canais.** A estrutura de
ViTs leves separa esses dois componentes. Na MobileNetV3, uma convolução 1×1 de
expansão e uma 1×1 de projeção fazem a interação entre canais, enquanto uma
convolução *depthwise* (DW) 3×3 realiza a fusão espacial. Como esses elementos
estão acoplados, movemos a DW para o início e posicionamos a camada SE opcional
logo após ela. Assim, separamos o misturador de tokens do de canais.

Também aplicamos reparametrização estrutural à camada DW para melhorar o
aprendizado. Na inferência, essa técnica elimina os custos computacionais e de
memória da conexão residual, o que favorece dispositivos móveis. Chamamos o
resultado de bloco RepViT. Nesse estágio, a latência cai para 0,81 ms, com uma
queda temporária da acurácia para 68,3%.

**Redução da razão de expansão e aumento da largura.** Em ViTs convencionais, a
razão de expansão do misturador de canais costuma ser 4, tornando a dimensão
oculta da FFN quatro vezes maior que a entrada. Como isso consome muito tempo de
inferência e há redundância entre canais, trabalhos recentes usam FFNs mais
estreitas. Na MobileNetV3-L, a razão varia de 2,3 a 6. No bloco RepViT, usamos
razão 2 em todos os estágios. A latência cai para 0,65 ms, permitindo dobrar os
canais após cada estágio para 48, 96, 192 e 384. Essas mudanças elevam a
acurácia a 73,5% com 0,89 ms. Aplicar apenas essas alterações ao bloco original
da MobileNetV3 produz resultado inferior: 73,0% com 0,91 ms.

**Figura 3 — Projeto do bloco.** A parte (a) mostra um bloco MobileNetV3 com SE
opcional. A parte (b) mostra o bloco RepViT, que separa os misturadores por
reparametrização estrutural. Normalização e não linearidade foram omitidas.

### 3.3. Projeto macro

**Convoluções iniciais no *stem*.** ViTs normalmente dividem a imagem em
*patches* não sobrepostos por uma convolução de kernel e passo grandes. Essa
operação pode prejudicar a otimização e tornar o treinamento sensível à receita.
Empilhar poucas convoluções 3×3 de passo 2 melhora estabilidade e desempenho.

A MobileNetV3-L usa um *stem* complexo na resolução mais alta, causando um
gargalo de latência e limitando o número inicial de filtros a 16. Substituímos
esse módulo por duas convoluções 3×3 com passo 2, com 24 e 48 filtros. A
latência cai para 0,86 ms e a acurácia sobe para 73,9%.

**Camadas de redução de resolução mais profundas.** ViTs geralmente usam uma
camada separada de mesclagem de *patches*, o que aumenta a profundidade e reduz
a perda de informação. A MobileNetV3-L reduz a resolução apenas por um gargalo
invertido com DW de passo 2. No RepViT, uma DW de passo 2 reduz o espaço, uma
convolução pontual 1×1 ajusta os canais, um bloco RepViT anterior aprofunda o
módulo e uma FFN posterior memoriza informação latente. O resultado é 75,4% de
acurácia com 0,96 ms.

**Classificador simples.** ViTs leves usam, em geral, *global average pooling*
seguido de uma camada linear. A MobileNetV3-L acrescenta uma convolução 1×1 e
uma camada linear para expandir as características. Como o último estágio do
RepViT já possui mais canais, substituímos esse classificador por sua versão
simples. A acurácia diminui 0,6 ponto, mas a latência cai para 0,77 ms.

**Proporção global entre estágios.** A razão de estágios descreve como os blocos
se distribuem pela rede. Mais blocos no terceiro estágio equilibram acurácia e
velocidade. Adotamos a razão 1:1:7:1 e aumentamos a profundidade para 2:2:14:2.
Isso produz 76,9% de acurácia com 0,91 ms.

**Figura 4 — Projeto macro.** Os pares (a)-(b), (c)-(d) e (e)-(f) mostram,
respectivamente, os projetos de *stem*, redução de resolução e classificador.
A RepViT possui quatro estágios nas resoluções H/4×W/4, H/8×W/8, H/16×W/16 e
H/32×W/32. C é a dimensão dos canais e B é o tamanho do lote.

### 3.4. Projeto micro

**Tamanho de kernel.** Kernels grandes podem capturar dependências de longo
alcance, mas são pouco adequados a dispositivos móveis devido ao custo de
computação e acesso à memória. Também recebem menos otimizações de compiladores
e bibliotecas que convoluções 3×3. Por isso, usamos convoluções 3×3 em todos os
módulos. A acurácia permanece em 76,9% e a latência cai para 0,89 ms.

**Posicionamento das camadas SE.** A autoatenção adapta seus pesos à entrada. As
camadas SE, como atenção por canais, compensam a falta desse atributo orientado
pelos dados nas convoluções. Contudo, também têm custo não desprezível.
Aplicamos SE alternadamente no primeiro, terceiro, quinto e demais blocos
ímpares de cada estágio. Assim maximizamos o ganho com pouco aumento de
latência: 77,4% de acurácia com 0,87 ms. Esse é o modelo RepViT final.

### 3.5. Arquitetura da rede

Construímos as variantes RepViT-M0.9, M1.0, M1.1, M1.5 e M2.3. O sufixo “MX”
indica latência de X ms no iPhone 12 com iOS 16. As variantes diferem no número
de canais e blocos em cada estágio; os detalhes completos estão no material
suplementar do artigo.

## 4. Experimentos

### 4.1. Classificação de imagens

Os experimentos usam ImageNet-1K com imagens 224×224 no treinamento e no
teste. Os modelos são treinados do zero por 300 ou 450 épocas com a mesma
receita. Para destilação, o professor é RegNetY-16GF com 82,9% top-1. A
latência, com lote 1, é medida no iPhone 12 após compilação pelo Core ML Tools.
RepViT-M0.9 é o resultado direto da modernização da MobileNetV3-L.

Com latências equivalentes, RepViT-M0.9 supera EfficientFormerV2-S0 e FastViT-T8
em 3,0 e 2,0 pontos de top-1. RepViT-M1.1 supera EfficientFormerV2-S1 em 1,7
ponto. RepViT-M1.0 ultrapassa 80% top-1 com 1,0 ms, e RepViT-M2.3 alcança 83,7%
com 2,3 ms. Isso mostra que CNNs puras e leves podem superar ViTs leves no
estado da arte ao incorporar seus projetos eficientes.

Sem destilação, a RepViT ainda supera os concorrentes nas diferentes faixas de
latência. A M1.0 ganha 2,7 pontos sobre MobileOne-S1 com 1,0 ms. A M2.3 supera
PoolFormer-S36 em 1,1 ponto e reduz a latência em 34,3%, de 3,5 para 2,3 ms.

**Tabelas 1 e 2.** A Tabela 1 compara parâmetros, GMACs, latência, vazão,
épocas e acurácia top-1 com destilação. A Tabela 2 apresenta resultados sem
destilação. Os valores numéricos completos estão nas páginas 6 e 7 do PDF.

### 4.2. RepViT encontra o SAM

O Segment Anything Model (SAM) apresenta forte transferência *zero-shot*, mas
seu custo é proibitivo em dispositivos móveis. Substituímos seu codificador de
imagem pelo RepViT-M2.3, criando o RepViT-SAM. Ele é treinado por oito épocas
nas mesmas condições do MobileSAM e usa apenas 1% do conjunto SAM-1B. A página
do projeto é <https://jameslahm.github.io/repvit-sam/>.

No iPhone 12, o RepViT-SAM executa normalmente, enquanto MobileSAM e ViT-B-SAM
ficam sem memória. No MacBook M1 Pro, ele é quase dez vezes mais rápido que o
MobileSAM. Também é avaliado em detecção de bordas *zero-shot* no BSDS500,
segmentação de instâncias *zero-shot* no COCO e no benchmark SegInW. Supera
MobileSAM e ViT-B-SAM em todos eles e obtém ODS e OIS comparáveis ao ViT-H-SAM,
que possui mais de 615 milhões de parâmetros.

**Tabela 3.** Latência em imagens 1024×1024: no iPhone, o codificador do
RepViT-SAM leva 48,9 ms e o decodificador de máscara, 11,6 ms; os concorrentes
ficam sem memória. No MacBook, RepViT-SAM leva 44,8 ms, MobileSAM 482,2 ms,
ViT-B-SAM 6249,5 ms e o decodificador 11,8 ms.

**Tabela 4.** Em detecção de bordas, segmentação de instâncias e SegInW, o
RepViT-SAM obtém, respectivamente, ODS 0,764, OIS 0,786, AP 0,773, AP 44,4 e AP
médio 46,1.

### 4.3. Tarefas posteriores

**Detecção de objetos e segmentação de instâncias.** Integramos RepViT ao
Mask R-CNN e realizamos experimentos no MS COCO 2017. Em tamanhos semelhantes,
RepViT supera os concorrentes em latência, AP de caixas e AP de máscaras.
RepViT-M1.1 supera o backbone EfficientFormer-L1 em 1,9 AP de caixas e 1,8 AP
de máscaras, com menor latência. RepViT-M1.5 é quase duas vezes mais rápida que
EfficientFormer-L3 com desempenho comparável. RepViT-M2.3 obtém AP de caixas
comparável e AP de máscaras superior a EfficientFormerV2-L com quase metade da
latência, evidenciando a vantagem das CNNs leves em tarefas de alta resolução.

**Segmentação semântica.** No ADE20K, integramos RepViT ao Semantic FPN.
RepViT-M1.1 supera EfficientFormer-L1 em 1,7 mIoU e é mais rápida. RepViT-M1.5
ganha 1,2 mIoU sobre EfficientFormerV2-S2 com quase 50% menos latência.
RepViT-M2.3 aumenta o mIoU em 0,9 e é quase duas vezes mais rápida que
EfficientFormerV2-L. Os resultados confirmam a RepViT como backbone geral.

**Tabela 5.** A tabela completa, na página 8 do PDF, reúne latência, AP de
caixas, AP de máscaras e mIoU para ResNet18, PoolFormer, PVT,
EfficientFormer/FormerV2 e RepViT.

### 4.4. Análises do modelo

**Reparametrização estrutural.** Remover, durante o treinamento, a topologia de
múltiplos ramos da reparametrização reduz consistentemente a acurácia das
variantes. Para M0.9, M1.5 e M2.3, os resultados sem/com reparametrização são
78,47/78,74%, 82,09/82,29% e 83,10/83,30%.

**Posicionamento de SE.** Comparamos remover SE, usá-la em todo bloco e usá-la
alternadamente. Na M0.9, os resultados são 77,92%/0,83 ms, 78,75%/0,92 ms e
78,74%/0,87 ms. Na M1.5, são 81,86%/1,48 ms, 82,29%/1,58 ms e 82,29%/1,52 ms.
A estratégia alternada oferece a melhor troca entre acurácia e latência.

## 5. Conclusão

Revisitamos o projeto eficiente de CNNs leves incorporando decisões
arquiteturais dos ViTs leves. O resultado, RepViT, é uma nova família de CNNs
para dispositivos móveis com recursos limitados. Ela supera ViTs e CNNs leves
do estado da arte em diversas tarefas, com boa relação entre desempenho e
latência. Isso evidencia o potencial das CNNs puras em dispositivos móveis.
Esperamos que RepViT sirva como uma base forte e inspire novas pesquisas em
modelos leves.

## 6. Agradecimentos

Este trabalho recebeu apoio do National Science and Technology Major Project
2022ZD0119401, da Beijing Natural Science Foundation (nº L223023) e da National
Natural Science Foundation of China (nºs 62271281, 61925107 e 62021002).

## Referências

As 79 referências bibliográficas foram mantidas em seu idioma e formato
originais nas páginas 9–12 do [artigo-fonte](2307.09283v8.pdf), evitando alterar
títulos oficiais, nomes de conferências e dados de publicação. As chamadas
numéricas desta tradução correspondem diretamente à lista original.
