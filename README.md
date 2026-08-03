# Seminários — TP558

Este repositório reúne os artigos e os materiais dos seminários da disciplina
TP558. Os trabalhos selecionados exploram arquiteturas eficientes para visão
computacional, desde detecção de objetos em tempo real até modelos leves para
dispositivos móveis.

## Artigos

| Seminário | Artigo | Tema |
| --- | --- | --- |
| 1 | [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](articles/sem1/2511.09554v2.pdf) | Detecção de objetos em tempo real e busca de arquiteturas neurais |
| 2 | [RepViT: Revisiting Mobile CNN From ViT Perspective](articles/sem2/2307.09283v8.pdf) | CNNs leves inspiradas em Vision Transformers |

## Seminário 1 — RF-DETR

**Título:** *RF-DETR: Neural Architecture Search for Real-Time Detection
Transformers*

**Autores:** Isaac Robinson, Peter Robicheaux, Matvei Popov, Deva Ramanan e
Neehar Peri.

**Publicação:** ICLR 2026.

O RF-DETR é uma família de detectores e segmentadores especializados que combina
pré-treinamento em larga escala com *Neural Architecture Search* (NAS). A busca
compartilha pesos entre diferentes sub-redes e permite encontrar configurações
com bons compromissos entre acurácia e latência sem retreinar cada arquitetura.

### Pontos principais

- Aplica NAS com compartilhamento de pesos ao detector completo, e não apenas ao
  *backbone*.
- Explora resolução de entrada, tamanho do *patch*, número de camadas do
  decodificador e quantidade de *queries*.
- Produz uma curva de Pareto de acurácia e latência a partir de um único
  treinamento no conjunto-alvo.
- Propõe um procedimento mais reprodutível para medir a latência de inferência.
- Estende a abordagem para segmentação de instâncias com o RF-DETR-Seg.

### Resultados destacados pelos autores

- O RF-DETR Nano alcança **48,0 AP no COCO**, superando o D-FINE Nano em 5,3 AP
  com latência semelhante.
- O RF-DETR 2x-Large supera o Grounding DINO Tiny em 1,2 AP no RF100-VL e
  executa aproximadamente 20 vezes mais rápido.
- O RF-DETR 2x-Large é apresentado como o primeiro detector em tempo real a
  ultrapassar 60 AP no COCO.

**Material:** [PDF local](articles/sem1/2511.09554v2.pdf) ·
[arXiv](https://arxiv.org/abs/2511.09554) ·
[código](https://github.com/roboflow/rf-detr)

## Seminário 2 — RepViT

**Título:** *RepViT: Revisiting Mobile CNN From ViT Perspective*

**Autores:** Ao Wang, Hui Chen, Zijia Lin, Jungong Han e Guiguang Ding.

O RepViT revisita o projeto de CNNs leves sob a perspectiva dos Vision
Transformers. Partindo do MobileNetV3-L, os autores incorporam progressivamente
decisões arquiteturais comuns em ViTs leves e chegam a uma família de CNNs puras
otimizada para dispositivos móveis.

### Pontos principais

- Separa o *token mixer* do *channel mixer* em uma estrutura semelhante ao
  MetaFormer.
- Emprega convoluções reparametrizáveis para manter a inferência eficiente.
- Reavalia o desenho dos blocos e decisões macro e micro da arquitetura.
- Mede latência diretamente em um iPhone 12, em vez de depender apenas de FLOPs
  ou do número de parâmetros.
- Avalia a arquitetura em classificação, detecção de objetos, segmentação
  semântica e segmentação de instâncias.

### Resultados destacados pelos autores

- O RepViT ultrapassa **80% de acurácia top-1 no ImageNet-1K** com latência de
  1,0 ms em um iPhone 12.
- O RepViT-M2.3 alcança 83,7% de acurácia top-1 com latência de 2,3 ms.
- O RepViT-SAM apresenta inferência aproximadamente 10 vezes mais rápida que o
  MobileSAM.

**Material:** [PDF local](articles/sem2/2307.09283v8.pdf) ·
[arXiv](https://arxiv.org/abs/2307.09283) ·
[código](https://github.com/THU-MIG/RepViT)

## Estrutura do repositório

```text
.
├── articles/
│   ├── sem1/
│   │   └── 2511.09554v2.pdf
│   └── sem2/
│       └── 2307.09283v8.pdf
├── LICENSE
└── README.md
```

## Licença

Os materiais autorais deste repositório são disponibilizados sob a
[licença MIT](LICENSE). Os artigos pertencem aos seus respectivos autores e
seguem as condições de distribuição de suas publicações originais.
