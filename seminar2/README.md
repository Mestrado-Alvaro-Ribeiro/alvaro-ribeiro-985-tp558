# Seminário 2 — TP558

Este diretório reúne o artigo e a reprodução reduzida do Seminário 2 da
disciplina TP558, dedicado ao RepViT e ao projeto de CNNs eficientes para
dispositivos móveis.

## Artigo

| Artigo | Tema |
| --- | --- |
| [RepViT: Revisiting Mobile CNN From ViT Perspective](2307.09283v8.pdf) | CNNs móveis, MetaFormer e reparametrização estrutural |

## Reprodução

O [notebook de reprodução](notebooks/repvit_reproduction.ipynb) implementa a
arquitetura RepViT-M0.9, realiza um treinamento reduzido no CIFAR-10, verifica
a equivalência numérica da reparametrização estrutural e compara latência,
parâmetros e resultados publicados.

A reprodução integral do artigo exige ImageNet-1K, 300 ou 450 épocas,
destilação e um iPhone 12 com Core ML. O notebook mantém essas diferenças
explícitas para não confundir o experimento didático com os números oficiais.

## Materiais

- [Artigo em PDF](2307.09283v8.pdf)
- [Resumo](repvit-resume.md)
- [Notebook de reprodução](notebooks/repvit_reproduction.ipynb)
- [Página no arXiv](https://arxiv.org/abs/2307.09283)
- [Código oficial do RepViT](https://github.com/THU-MIG/RepViT)

## Licença

Os materiais autorais deste diretório são disponibilizados sob a
[licença MIT](../LICENSE). O artigo pertence aos seus respectivos autores e
segue as condições de distribuição de sua publicação original.
