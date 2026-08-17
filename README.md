# Seminário 1 — TP558

Este repositório reúne o artigo e os materiais do Seminário 1 da disciplina
TP558, dedicado ao RF-DETR e à detecção de objetos em tempo real com busca de
arquiteturas neurais.

## Artigo

| Seminário | Artigo | Tema |
| --- | --- | --- |
| 1 | [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](articles/sem1/2511.09554v2.pdf) | Detecção de objetos em tempo real e busca de arquiteturas neurais |

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
[resumo](seminar1/rf-detr-resume.md) ·
[notebook experimental](seminar1/notebooks/rf_detr_reproduction.ipynb) ·
[arXiv](https://arxiv.org/abs/2511.09554) ·
[código](https://github.com/roboflow/rf-detr)

## Estrutura do repositório

```text
.
├── seminar1/
│   ├── 2511.09554v2.pdf
│   ├── notebooks/
│   │   └── rf_detr_reproduction.ipynb
│   └── rf-detr-resume.md
├── LICENSE
└── README.md
```

## Licença

Os materiais autorais deste repositório são disponibilizados sob a
[licença MIT](LICENSE). Os artigos pertencem aos seus respectivos autores e
seguem as condições de distribuição de suas publicações originais.
