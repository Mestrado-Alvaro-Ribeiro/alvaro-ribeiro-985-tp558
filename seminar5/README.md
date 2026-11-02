# Seminário 5 — Machine learning runtimes e otimizadores

Materiais de TP558 sobre implantação e otimização da inferência: ONNX e ONNX
Runtime, LiteRT (antigo TensorFlow Lite), TensorRT, OpenVINO, TVM e JAX/XLA.

## Materiais

- [Resumo e análise em português](ml-runtimes-resume.md)
- [Apresentação Inatel em PPTX](ML_Runtimes_Apresentacao_Inatel.pptx)
- [Apresentação Inatel em PDF](ML_Runtimes_Apresentacao_Inatel.pdf)
- [Notebook didático de comparação em CPU](notebooks/runtimes_comparison.ipynb)
- [Dependências do notebook](notebooks/requirements.txt)

## Artigos

| Artigo | Arquivo | Papel no seminário |
| --- | --- | --- |
| Suchý e Turčaník (2026), *Review of large YOLOv8 and RT-DETR energy efficiency on edge devices for real-time detection* | [PDF](s41598-026-46453-6.pdf) · [Publicação](https://www.nature.com/articles/s41598-026-46453-6) | Acurácia, latência, vídeo e energia em Raspberry Pi 5 e Jetson Orin NX |
| Ratul, Zhou e Yang (2025), *Accelerating Deep Learning Inference: A Comparative Analysis of Modern Acceleration Frameworks* | [PDF](electronics-14-02977.pdf) · [Publicação](https://www.mdpi.com/2079-9292/14/15/2977) | PyTorch, ONNX Runtime, TensorRT, TVM e JAX na Jetson AGX Orin |
| Gomez Fernandez et al. (2026), *Benchmarking Edge Inference Strategies for Deep Learning Models in Industrial Machine Vision* | [PDF v1](2607.11356v1.pdf) · [arXiv](https://arxiv.org/abs/2607.11356v1) | Comparação de runtimes entre CPUs, GPUs, CNNs e Grounding DINO |

Os resultados dos artigos são apresentados separadamente: os dispositivos,
modelos, precisões, lotes e limites da medição diferem entre os estudos.

## Demonstração prática

O notebook constrói uma CNN pequena com pesos aleatórios fixos, exporta para
ONNX e compara PyTorch, ONNX Runtime com otimizações desabilitadas/habilitadas
e OpenVINO em **CPU**. Verifica as saídas em várias entradas antes de medir
latência, registra versões, hardware, threads, aquecimento, média, desvio
padrão, p50 e p95. A sessão e a compilação são medidas fora da inferência.

A demonstração não requer dataset, checkpoint ou treinamento. Os pesos
aleatórios permitem estudar execução e equivalência numérica, mas **não
avaliam acurácia** de uma tarefa. Não reproduz os números dos artigos, não
mede energia e não executa TensorRT, LiteRT ou aceleradores externos.
Esses caminhos são discutidos nos slides e ao final do notebook.

Para executar com Python 3.11 ou superior:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r seminar5/notebooks/requirements.txt
python -m ipykernel install --user --name seminar5 --display-name "Seminário 5"
```

Abra o notebook em Jupyter ou VS Code, selecione esse kernel e execute todas
as células. A instalação inicial exige internet. Os modelos e resultados
gerados ficam em `notebooks/outputs/`, ignorado pelo Git. As saídas incorporadas
no notebook documentam uma execução de referência e variam entre máquinas.

## Documentação oficial

- [ONNX: representação de modelos](https://onnx.ai/onnx/)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [LiteRT](https://developers.google.com/edge/litert)
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [OpenVINO](https://docs.openvino.ai/)
- [Apache TVM](https://tvm.apache.org/docs/)
- [JAX e compilação JIT](https://docs.jax.dev/en/latest/jit-compilation.html)

## Licença

Os materiais autorais seguem a [licença MIT](../LICENSE). A identidade visual
Inatel é reutilizada do modelo dos seminários anteriores. Os PDFs pertencem
a seus autores: Scientific Reports, **CC BY-NC-ND 4.0**; Electronics,
**CC BY 4.0**; arXiv v1, **CC BY 4.0**. A licença MIT não se aplica aos artigos
nem às marcas. As tabelas e os gráficos didáticos citam a origem dos dados.
