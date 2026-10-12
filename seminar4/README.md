# Seminário 4 — D-FINE

Materiais do Seminário 4 de TP558 sobre detecção de objetos em tempo real com
D-FINE, de Peng et al. (ICLR 2025 Spotlight).

## Materiais

- [Artigo original (arXiv v1)](2410.13842v1.pdf)
- [Resumo e análise em português](d-fine-resume.md)
- [Apresentação em PPTX](D-FINE_Apresentacao_Inatel.pptx)
- [Apresentação em PDF](D-FINE_Apresentacao_Inatel.pdf)
- [Notebook didático de inferência](notebooks/dfine_inferencia.ipynb)
- [Vídeo Manim: FDR da query à caixa](video/outputs/fdr_explicado.mp4)

## Reprodução didática

O notebook usa o checkpoint público `ustc-community/dfine-nano-coco` da
Hugging Face e uma imagem do COCO. Ele mostra as detecções finais, acompanha
uma query pelas camadas do decodificador e visualiza as distribuições de
correção das quatro bordas (FDR). Ele compara os papéis de FDR e GO-LSD e
mostra, como diagnóstico, a divergência entre distribuições de camadas iniciais
e final. A Tabela 3 do artigo apresenta a contribuição dos ajustes de
eficiência, FDR e GO-LSD para o D-FINE-L. O notebook **não** reproduz a perda
GO-LSD completa, o treinamento, o AP no COCO ou a latência em TensorRT dos
autores.
A primeira execução exige internet para instalar dependências e baixar modelo
e imagem. Uma CPU pode executar o exemplo, com latência diferente da reportada
no artigo.

## Fontes

- [Artigo no arXiv](https://arxiv.org/abs/2410.13842)
- [Código e checkpoints oficiais](https://github.com/Peterande/D-FINE)
- [Documentação no Transformers](https://huggingface.co/docs/transformers/model_doc/d_fine)
- [Checkpoint Nano usado no notebook](https://huggingface.co/ustc-community/dfine-nano-coco)

O artigo pertence aos autores e é distribuído conforme a [licença indicada no
arXiv](https://arxiv.org/html/2410.13842v1). Os materiais autorais seguem a
[licença MIT](../LICENSE).
