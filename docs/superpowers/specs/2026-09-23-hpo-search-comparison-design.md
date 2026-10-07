# Notebook de Comparação de HPO — Design

## Objetivo

Criar um notebook curto e reproduzível para o Seminário 3 que compare Grid
Search, Random Search e TPE em uma mesma função-objetivo simulada.

## Escopo

- Usar três hiperparâmetros ilustrativos: `learning_rate`, `dropout` e
  `width`.
- Avaliar uma função de perda simulada, com ruído determinístico, em vez de
  treinar uma rede. O notebook deve executar em segundos.
- Usar o mesmo orçamento aproximado e uma semente fixa para todos os métodos.
- Implementar a grade diretamente e executar Random Search e TPE com Optuna.
- Mostrar a melhor configuração, a melhor perda e a curva da melhor perda
  acumulada por avaliação.
- Explicar que o experimento ilustra os conceitos dos slides, mas não reproduz
  os benchmarks dos artigos nem BOHB/Auto-PyTorch.

## Arquitetura

O notebook será o único artefato executável novo, em
`seminar3/notebooks/hpo_search_comparison.ipynb`. Células de preparação definem
as dependências, constantes e a função simulada. Células seguintes executam
cada estratégia, consolidam o histórico de avaliações em uma tabela e geram os
gráficos. A conclusão interpreta os resultados sem afirmar superioridade
universal do TPE.

## Restrições

- Não adicionar dependências além de `numpy`, `pandas`, `matplotlib` e
  `optuna`.
- Semente fixa para reprodutibilidade.
- Sem download de datasets, treinamento de modelos ou GPU.
- O notebook deve informar como instalar as dependências caso elas não estejam
  disponíveis.

## Verificação

- Executar todas as células do notebook do início ao fim em um ambiente limpo.
- Confirmar que os três métodos geram registros, tabela final e gráfico de
  convergência.
- Confirmar que a explicação distingue TPE de BOHB/Auto-PyTorch.
