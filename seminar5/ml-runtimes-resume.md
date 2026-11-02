# Machine learning runtimes e otimizadores de inferência

## 1. Problema e ideia central

Uma rede treinada define uma função, mas não determina uma única forma de
executá-la. O mesmo modelo pode produzir tempos de resposta diferentes quando
mudam o motor de inferência, a precisão numérica, o dispositivo ou a forma de
transferir os dados. A implantação exige escolher uma combinação que satisfaça
requisitos de qualidade, latência, memória, energia e integração.

Os três artigos deste seminário são estudos empíricos de implantação. Ratul,
Zhou e Yang comparam cinco frameworks na Jetson AGX Orin. Suchý e Turčaník
avaliam detectores grandes em Raspberry Pi 5 e Jetson Orin NX, incluindo
consumo elétrico e processamento de vídeo. Gomez Fernandez et al. ampliam a
comparação entre arquiteturas e plataformas e mostram um caso em que TensorRT
perde para PyTorch em GPUs de desktop. Em conjunto, sustentam que a escolha
de runtime precisa ser medida no sistema-alvo, sem estabelecer um vencedor
universal.

## 2. Vocabulário: formato, runtime, compilador e backend

**ONNX** é uma representação de grafos computacionais com operadores e versões
definidas por *opsets*. Um arquivo `.onnx` descreve entradas, operações, pesos
e saídas. Ele permite intercâmbio, mas não garante que qualquer runtime
execute todos os operadores ou produza o mesmo desempenho.

**ONNX Runtime** é um motor de execução. Pode otimizar o grafo e delegar
subgrafos a *Execution Providers*, como CPU, CUDA, TensorRT e OpenVINO. Portanto,
comparar ONNX Runtime com TensorRT exige informar qual provider está ativo.
O formato ONNX e o runtime de mesmo nome ocupam funções diferentes.

Um **compilador ou otimizador de inferência** transforma o grafo para um alvo,
selecionando kernels, layouts e estratégias de memória. Um **runtime** carrega
o artefato e coordena sua execução. Ferramentas como TensorRT e OpenVINO
reúnem preparação e execução. **Backend** designa a implementação usada para
executar operadores ou subgrafos em determinado hardware.

Neste tema, “otimizador” significa otimização da execução. SGD e Adam ajustam
os pesos no treinamento e têm outra finalidade.

## 3. Principais ferramentas

| Ferramenta | Papel | Alvos e mecanismo | Aspecto que precisa ser verificado |
| --- | --- | --- | --- |
| ONNX | Representação de modelos | Grafo com operadores, tensores e opsets | Exportação, formas dinâmicas e suporte a operadores |
| ONNX Runtime | Runtime com providers | CPU e aceleradores, conforme o provider e a instalação | Particionamento do grafo, fallback e movimentação de dados |
| LiteRT | Runtime do ecossistema Google AI Edge, anteriormente TensorFlow Lite | Execução local, kernels e aceleração por APIs/delegates compatíveis | Conversão, operadores e suporte efetivo ao acelerador |
| TensorRT | Otimizador e runtime da NVIDIA | Compilação e seleção de kernels para GPUs NVIDIA | Compatibilidade do engine, formas, precisão e versões |
| OpenVINO | Conversão, otimização e runtime | CPU e dispositivos compatíveis, incluindo GPU/NPU Intel | Plugin, dispositivo escolhido e precisão interna |
| Apache TVM | Infraestrutura de compilação de ML | Transformações de grafo e geração/ajuste de código para alvos | Cobertura de operadores e custo de tuning |
| JAX/XLA | Programação numérica e compilação JIT | Compilação de funções e otimização conforme o backend | Compilação inicial, formas e execução assíncrona |

LiteRT é o nome atual do ecossistema anteriormente chamado TensorFlow Lite.
Os resultados rotulados como **TFLite** no artigo de 2026 correspondem à pilha
testada pelos autores, não a uma avaliação automática das versões atuais do
LiteRT. NCNN, MNN, Paddle Lite, HailoRT e Edge TPU aparecem como alternativas
ou caminhos de implantação no estudo energético.

As funções das ferramentas foram conferidas na documentação oficial listada
ao final. O suporte exato depende da versão e do dispositivo.

## 4. O que pode ser otimizado

### Transformações do grafo e kernels

*Constant folding* calcula antecipadamente operações cujas entradas são
constantes. Eliminar nós redundantes reduz trabalho. A fusão pode combinar,
por exemplo, convolução e ativação em um kernel, reduzindo lançamentos e
acessos intermediários à memória. A escolha do algoritmo de convolução e do
layout dos tensores também influencia o desempenho.

A fusão depende do padrão reconhecido, das formas e do backend. A presença
de uma opção de otimização não prova que houve fusão nem que a latência caiu.
É necessário inspecionar o grafo ou o perfil e medir o resultado.

### Memória e comunicação

O planejamento de memória reaproveita buffers quando seus tempos de vida não
se sobrepõem. Evitar cópias e manter tensores no acelerador pode ser tão
relevante quanto acelerar uma operação. Um operador sem suporte pode causar
fallback para CPU e transferências adicionais entre dispositivos.

### Precisão e quantização

FP16 reduz a representação em relação a FP32 e pode usar unidades de cálculo
mais rápidas quando o hardware e o kernel permitem. A quantização inteira
aproxima valores reais por inteiros, por exemplo:

$$
x \approx s(q-z),
$$

em que $s$ é a escala e $z$, o ponto zero. A quantização pós-treinamento pode
exigir calibração representativa das ativações. O treinamento consciente de
quantização incorpora essa aproximação ao treinamento.

INT8 não garante menor latência nem a preservação de acurácia. Conversões,
operadores parcialmente quantizados e falta de kernels adequados podem
reduzir o benefício. A saída deve ser validada após exportação e novamente
após quantização, para separar as duas fontes de mudança.

### Lote, formas e compilação

Lotes maiores podem elevar throughput, mas aumentam memória e podem aumentar
o tempo de espera por uma resposta. Especializar formas reduz a flexibilidade
e pode facilitar otimizações. A compilação e o tuning têm custo inicial que
deve ser separado do tempo de execução em regime estável.

## 5. Métricas e protocolo de medição

Para uma sequência serial, sem sobreposição entre etapas, o tempo de aplicação
pode ser decomposto como:

$$
T_{\mathrm{aplicação}} = T_{\mathrm{entrada}} + T_{\mathrm{pré}} +
T_{\mathrm{transferências}} + T_{\mathrm{rede}} + T_{\mathrm{pós}}.
$$

Pipelines concorrentes podem sobrepor etapas. Nesse caso, a soma não determina
diretamente throughput. Mesmo em lote 1, $1000/T_{\mathrm{rede}}$ não representa
automaticamente os FPS de vídeo da aplicação. Para um lote de $B$ amostras
processado serialmente em $t_B$ milissegundos, o throughput daquele trecho é
$1000B/t_B$ amostras por segundo.

Potência média $P$ é medida em watts. Energia por imagem é aproximadamente
$P/\mathrm{FPS}$, em joules por imagem, quando potência e throughput se referem
ao mesmo intervalo e sistema. FPS/W expressa imagens por joule. Menor potência
pode acompanhar maior energia por imagem se a execução durar muito mais.

Um protocolo útil informa modelo e pesos, formato de entrada, precisão, lote,
threads, hardware, versões, provider e limites da medição. Deve separar carga
e compilação, aquecer o runtime, sincronizar dispositivos assíncronos e
registrar dispersão. Média e desvio padrão descrevem a amostra, enquanto p50
e p95 ajudam a observar a distribuição das latências. Amostras repetidas na
mesma sessão não substituem replicações independentes.

## 6. Artigo 1 — eficiência energética de YOLOv8 e RT-DETR

**Referência:** Ivan Suchý e Michal Turčaník. *Review of large YOLOv8 and
RT-DETR energy efficiency on edge devices for real-time detection*.
Scientific Reports, 16, 10908, 2026.
[DOI e texto](https://www.nature.com/articles/s41598-026-46453-6).

### Objetivo e método

O estudo compara YOLOv8l e RT-DETR-l pré-treinados em COCO. A acurácia é
avaliada no COCO val2017. FPS e energia são medidos no vídeo MOT16-13, com
750 quadros de 1920 × 1080 a 25 FPS. Os alvos são Raspberry Pi 5 de 8 GB,
com caminhos Coral Edge TPU e Hailo-8, e Jetson Orin NX de 16 GB em modo MAXN.
Não se trata da mesma Jetson AGX Orin dos outros artigos.

A potência considera a plataforma completa e os aceleradores externos.
O artigo distingue tempo da rede, exposto pelo runtime, de FPS da cadeia
de vídeo. Define tempo real como processamento sustentado de pelo menos
25 FPS. Nenhuma configuração testada atende a esse critério.

### Resultados selecionados

As Tabelas 6 e 7 fornecem os seguintes resultados na Jetson Orin NX:

| Modelo | Runtime | Rede (ms) | Vídeo (FPS) | Potência (W) | FPS/W |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8l | PyTorch | 117 | 8,21 | 19,17 | 0,4283 |
| YOLOv8l | TensorRT FP32 | 91 | 10,46 | 20,39 | 0,5130 |
| YOLOv8l | TensorRT INT8 | 55 | 15,94 | 16,00 | 0,9960 |
| RT-DETR-l | PyTorch | 148 | 6,52 | 21,43 | 0,3043 |
| RT-DETR-l | TensorRT FP32 | 77 | 12,17 | 20,29 | 0,5999 |
| RT-DETR-l | TensorRT INT8 | 48 | 18,01 | 17,00 | 1,0597 |

A ordenação dos dois detectores muda com o runtime: YOLOv8l é mais rápido
com PyTorch, enquanto RT-DETR-l apresenta maior FPS com TensorRT nessa
configuração. O RT-DETR-l INT8 consome aproximadamente 0,944 J por quadro,
conforme a Tabela 8. Esse valor se refere à cadeia e à plataforma medidas.

Na Raspberry Pi 5, YOLOv8l com HailoRT alcança **8,17 FPS e 1,7281 FPS/W**,
contra **6,89 FPS e 1,3172 FPS/W** no caminho Edge TPU. Contudo, Edge TPU usa
entrada **224 × 224**, enquanto as demais configurações reportadas usam
640 × 640, salvo indicação contrária. Na Tabela 3, seu mAP50–95 é 0,324,
contra 0,530 em PyTorch e 0,541 em HailoRT. A diferença envolve caminho de
implantação, precisão e resolução, impedindo atribuição apenas ao acelerador.

### Leitura crítica

O artigo oferece evidência de sistema, incluindo energia e vídeo, e evidencia
o custo de conversão e compatibilidade. RT-DETR não foi implantado em Edge
TPU e Hailo-8 por limitações de conversão e operadores. Isso constitui um
resultado de viabilidade da pilha testada, não uma impossibilidade geral para
qualquer implementação futura de Transformers.

As Tabelas 2 e 3 têm títulos que mencionam validação no Colab, enquanto o
texto também descreve verificações nos dispositivos. Essa apresentação deixa
ambígua a localização exata de parte das avaliações de acurácia. Além disso,
os indicadores por GFLOP são normalizações descritivas: FLOPs não capturam
tráfego de memória, composição dos operadores ou utilização real do hardware.

## 7. Artigo 2 — comparação de cinco frameworks na Jetson AGX Orin

**Referência:** Ishrak Jahan Ratul, Yuxiao Zhou e Kecheng Yang.
*Accelerating Deep Learning Inference: A Comparative Analysis of Modern
Acceleration Frameworks*. Electronics, 14(15), 2977, 2025.
[DOI e texto](https://www.mdpi.com/2079-9292/14/15/2977).

### Objetivo e método

O artigo compara PyTorch, ONNX Runtime, TensorRT, Apache TVM e JAX na Jetson
AGX Orin. Inclui ResNet-152, MobileNetV2, SqueezeNet, EfficientNet-B0, VGG16,
Swin Transformer e YOLOv5s. A metodologia descreve ImageNet, entradas
224 × 224, FP16, 50 aquecimentos e lotes de 2 a 128. JAX é limitado aos
lotes 2, 4 e 8 por memória. ONNX Runtime usa o provider de GPU, segundo os
autores. O baseline PyTorch inclui TorchScript e otimizações cuDNN, portanto
não representa necessariamente execução eager sem preparação.

### Resultados selecionados

Tempos médios por amostra, em milissegundos, da Tabela 6:

| Runtime | ResNet-152 | MobileNet | VGG16 | Swin |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 9,239 | 4,201 | 1,713 | 7,267 |
| ONNX Runtime | 285,534 | 54,402 | 158,146 | 19,813 |
| TensorRT | 2,282 | 1,136 | 2,195 | 3,949 |
| TVM | 7,429 | 9,529 | 8,908 | 5,275 |
| JAX | 29,096 | 11,235 | 19,326 | 22,268 |

TensorRT apresenta o menor valor em seis das sete colunas da tabela
completa. A exceção é **VGG16**, em que **PyTorch (1,713 ms)** é mais rápido
que **TensorRT (2,195 ms)**. Essa exceção deve ser preservada, embora o texto
do artigo descreva TensorRT como o mais rápido em todos os modelos.

O throughput máximo, apresentado separadamente na Tabela 7, não acompanha
necessariamente a mesma ordenação. Em Swin, PyTorch registra 1.250,40
amostras/s e TensorRT 317,52. Uma conclusão sobre menor tempo por amostra
não estabelece maior throughput para todos os lotes.

### Leitura crítica

O estudo cobre várias dimensões de custo, mas seus números precisam ser
interpretados com as configurações de conversão e lote. A Tabela 6 resume
tempo por amostra sem tornar explícito, em cada célula, o lote associado.
Ela não deve ser usada como uma tabela de latência de requisições em lote 1.

O desempenho baixo de ONNX Runtime descreve a implantação desse estudo.
Não demonstra que o runtime será lento em qualquer plataforma ou provider.
As diferenças de acurácia entre frameworks exigem verificar pesos,
pré-processamento, precisão e semântica dos operadores. A otimização do grafo
não implica, por si só, melhora da capacidade preditiva.

Há inconsistências de apresentação: a metodologia anuncia seis modelos e
enumera sete; as tabelas de Top-1/Top-5 incluem YOLOv5s sem esclarecer
suficientemente a adaptação do detector à avaliação descrita. Essas colunas
não devem ser interpretadas como AP de detecção. Potência média menor também
não demonstra menor energia por imagem sem considerar o tempo de execução.

## 8. Artigo 3 — runtimes em visão industrial e hardware heterogêneo

**Referência:** Miguel Gomez Fernandez, David Castro Boga, Roi Mendez-Rial e
Eric Lopez-Lopez. *Benchmarking Edge Inference Strategies for Deep Learning
Models in Industrial Machine Vision*. arXiv:2607.11356v1, 13 julho 2026.
[Texto e versão](https://arxiv.org/html/2607.11356v1).

### Objetivo e método

Os autores comparam PyTorch, ONNX Runtime, OpenVINO e TensorRT. Avaliam
YOLOv8 com entrada 640 × 640, UNet com backbone ResNeXt50 e entrada
1024 × 1024, e Grounding DINO com entrada 1200 × 800 e prompt `dog`.

O protocolo usa FP32 declarado, lote 1 e 1.000 execuções, descartando as
primeiras dez e calculando média e desvio padrão das 990 restantes.
A temporização de GPU inclui sincronização. Pré e pós-processamento ficam
fora da medição. Modos de aceleração padrão dos frameworks não são
desativados manualmente, de modo que FP32 declarado não equivale a um
controle completo de todos os modos aritméticos internos.

Há três desktops com CPUs Intel e cinco GPUs NVIDIA ao todo, contando a GPU
da Jetson AGX Orin. OpenVINO é testado apenas em CPU. As diferenças entre
máquinas não isolam causalmente o efeito de cada componente de hardware.

### Resultados selecionados

Na Tabela II, para YOLOv8:

| Alvo | PyTorch (ms) | ONNX Runtime (ms) | OpenVINO (ms) | TensorRT (ms) |
| --- | ---: | ---: | ---: | ---: |
| Intel i9-9820X, CPU | 28,7 | 24,1 | 10,9 | — |
| Jetson AGX Orin, CPU | 470 | 184 | 429,2 | — |
| Quadro RTX 6000, GPU | 7,780 | 3,35 | — | 1,55 |
| Jetson AGX Orin, GPU | 20,84 | 19,69 | — | 10,57 |

OpenVINO apresenta os menores tempos nos três desktops Intel e nos três
modelos avaliados. Na CPU ARM da Jetson, ONNX Runtime apresenta os menores
tempos. TensorRT vence nos casos de CNN em GPU, incluindo YOLOv8 e UNet.

Grounding DINO modifica a ordenação. Na RTX 2080 Ti, a Tabela IV informa
**133,5 ms em PyTorch**, **250,4 ms em ONNX Runtime** e **268 ms em TensorRT**.
Nos quatro casos de GPU de desktop, TensorRT é aproximadamente 1,6 a 2,1
vezes mais lento que PyTorch. Na GPU da Jetson, porém, TensorRT obtém
1.082,7 ms, contra 1.290,8 ms em PyTorch. A exceção também depende do alvo.

### Leitura crítica

O estudo controla lote e escopo da temporização e inclui uma arquitetura
visuolinguística. Não mede consumo elétrico nem throughput de aplicação e
não apresenta uma avaliação de acurácia pós-conversão que permita concluir
equivalência preditiva entre todos os caminhos.

Os resultados se referem às versões registradas, incluindo TensorRT 8.6.1
nos desktops e 8.5.2 na Jetson. Não são uma comparação de todas as versões
atuais. Estrutura do grafo exportado e entradas dinâmicas são explicações
plausíveis discutidas pelos autores para Grounding DINO, mas o benchmark
não isola experimentalmente a contribuição de cada mecanismo.

## 9. Comparação dos estudos

| Aspecto | Suchý e Turčaník | Ratul et al. | Gomez Fernandez et al. |
| --- | --- | --- | --- |
| Hardware | Pi 5 e Orin NX | Jetson AGX Orin | Desktops Intel/NVIDIA e AGX Orin |
| Precisão | FP32, INT8 e caminhos específicos | FP16 declarado | FP32 declarado |
| Lote | Pipeline de vídeo, conforme a implantação | 2–128, JAX até 8 | 1 |
| Tempo reportado | Rede e FPS ponta a ponta separados | Tempo por amostra e throughput máximo | Apenas rede |
| Qualidade | mAP no COCO val2017 | Top-1/Top-5 com ressalvas de apresentação | Sem avaliação de qualidade pós-exportação |
| Energia | Potência total, FPS/W, J/quadro | Potência e memória | Não medida |
| Principal contribuição para a aula | Avaliar a cadeia e o custo energético | Discutir lote e diferentes dimensões de custo | Evidenciar dependência de arquitetura e hardware |

Os tempos não devem ser reunidos em um ranking entre artigos. Mesmo quando
dois trabalhos usam uma Jetson AGX Orin, modelos, formas, precisão, versões e
limites de medição continuam diferentes.

## 10. Demonstração e conexão com os seminários anteriores

RF-DETR, RepViT e D-FINE discutem escolhas arquiteturais e de treinamento.
O seminário de HPO discute a configuração dessas escolhas. Este seminário
acrescenta a etapa de execução: o ganho arquitetural precisa sobreviver à
exportação e encontrar kernels adequados no dispositivo-alvo. A
reparametrização do RepViT é um exemplo de transformação voltada ao grafo de
inferência. A latência TensorRT/T4 do D-FINE ilustra por que o protocolo deve
acompanhar qualquer número de velocidade.

O notebook deste diretório usa uma CNN pequena com pesos aleatórios fixos,
sem pretensão de resolver uma tarefa visual. Compara PyTorch e o mesmo grafo
ONNX em dois níveis de otimização do ONNX Runtime, além de OpenVINO em CPU.
Verifica equivalência em várias entradas e mede preparação e inferência
separadamente, com uma thread por runtime e precisão FP32 configurada.

Os resultados são locais. Não estimam AP, energia ou desempenho de TensorRT
e LiteRT. Uma continuação experimental seria substituir a CNN por um modelo
treinado, verificar sua métrica no dataset e repetir no hardware de destino,
medindo também transferências, pré-processamento e pós-processamento.

## 11. Síntese

A implantação combina representação, compilação e execução. Nos protocolos
dos artigos, TensorRT beneficia várias CNNs em GPU, OpenVINO se destaca em
CPUs Intel e ONNX Runtime fornece um caminho competitivo em outros alvos.
Grounding DINO e VGG16 expõem exceções que impedem transformar essas
observações em regras universais.

A decisão deve partir do requisito da aplicação e preservar a qualidade
após conversão. Em seguida, mede-se a combinação de modelo, runtime e
hardware com escopo definido. Os três estudos delimitam essa conclusão:
seus resultados não substituem um benchmark no dispositivo e na carga de
trabalho pretendidos.

## Fontes

1. [Suchý e Turčaník, Scientific Reports (2026)](https://www.nature.com/articles/s41598-026-46453-6), especialmente Tabelas 3–9 e protocolo experimental.
2. [Ratul, Zhou e Yang, Electronics (2025)](https://www.mdpi.com/2079-9292/14/15/2977), especialmente seção 4 e Tabelas 6 e 7.
3. [Gomez Fernandez et al., arXiv v1 (2026)](https://arxiv.org/html/2607.11356v1), especialmente seção II-D e Tabelas II–IV.
4. [ONNX](https://onnx.ai/onnx/) e [ONNX Runtime](https://onnxruntime.ai/docs/).
5. [LiteRT](https://developers.google.com/edge/litert).
6. [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/).
7. [OpenVINO](https://docs.openvino.ai/).
8. [Apache TVM](https://tvm.apache.org/docs/) e [JAX JIT](https://docs.jax.dev/en/latest/jit-compilation.html).
