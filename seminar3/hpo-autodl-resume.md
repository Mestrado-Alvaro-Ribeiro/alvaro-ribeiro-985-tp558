# Otimização de hiperparâmetros, Auto-PyTorch e Optuna

## Visão geral

Este texto resume separadamente os artigos **Algorithms for Hyper-Parameter
Optimization**, de Bergstra et al. (2011), e **Auto-PyTorch: Multi-Fidelity
MetaLearning for Efficient and Robust AutoDL**, de Zimmer, Lindauer e Hutter
(2021). Em seguida, apresenta o Optuna como ferramenta prática e compara as
três contribuições.

Os trabalhos ocupam níveis diferentes. Bergstra et al. estudam algoritmos para
escolher hiperparâmetros. O Auto-PyTorch usa essa otimização dentro de um
sistema AutoDL que também escolhe arquiteturas e forma *ensembles*. O Optuna é
uma biblioteca geral para implementar estudos de otimização, mas não é o
algoritmo empregado nos experimentos do Auto-PyTorch.

---

## 1. Algorithms for Hyper-Parameter Optimization

### Problema e objetivo

O treinamento determina os parâmetros internos de um modelo, como os pesos de
uma rede neural, mas depende de hiperparâmetros definidos externamente, como
taxa de aprendizado, regularização, número de camadas e quantidade de unidades.
O artigo formula sua escolha como a minimização de uma função de validação cara
e sem gradiente diretamente disponível:

$$
x^* = \arg\min_{x \in \mathcal{X}} f(x),
$$

em que $x$ é uma configuração e $f(x)$, sua perda de validação. Como cada
avaliação pode exigir o treinamento completo de um modelo, o objetivo é
encontrar configurações boas com poucas tentativas.

O trabalho compara busca em grade, busca aleatória e dois métodos sequenciais
baseados em *expected improvement* (EI): um com processos gaussianos e outro
com **Tree-structured Parzen Estimator (TPE)**. Os experimentos envolvem redes
neurais e *deep belief networks*.

### Busca aleatória contra busca em grade

A busca em grade testa valores predeterminados de cada hiperparâmetro. Ela
desperdiça avaliações quando somente poucas dimensões influenciam o resultado,
pois repete os mesmos valores dessas dimensões importantes enquanto varia
parâmetros pouco relevantes.

A busca aleatória amostra cada configuração independentemente. Sob o mesmo
orçamento, tende a experimentar mais valores distintos nas dimensões que
realmente importam. Também é simples, paralelizável e aceita distribuições
adequadas para cada parâmetro. O artigo a estabelece como um baseline mais
forte do que a grade. Entretanto, ela não aprende com resultados anteriores e
pode continuar explorando regiões que já parecem ruins.

### Otimização sequencial e TPE

Na otimização sequencial baseada em modelos, o histórico de pares
$(x, f(x))$ orienta a próxima configuração. Um modelo substituto representa o
comportamento observado, e uma função de aquisição equilibra exploração e
aproveitamento. O EI mede a melhoria esperada sobre o melhor valor conhecido:

$$
x_{t+1} = \arg\max_x EI(x).
$$

O TPE inverte a modelagem bayesiana usual. Em vez de estimar diretamente
$p(y \mid x)$, modela $p(x \mid y)$ e divide as observações por um limiar
$y^*$:

$$
p(x \mid y) =
\begin{cases}
l(x), & y < y^*,\\
g(x), & y \geq y^*.
\end{cases}
$$

Aqui, $l(x)$ descreve configurações boas e $g(x)$, as demais. A seleção
favorece candidatos com alta razão $l(x)/g(x)$, relacionada à maximização do
EI. Sua estrutura em árvore representa naturalmente espaços condicionais, nos
quais a escolha de um algoritmo ativa apenas os hiperparâmetros pertinentes.

### Resultados e leitura crítica

Os experimentos mostram que a busca aleatória é competitiva e pode superar
grades manuais. Nos problemas mais difíceis de otimização de *deep belief
networks*, os métodos sequenciais encontram configurações melhores e com maior
eficiência do que as estratégias não adaptativas estudadas.

A contribuição duradoura é dupla: justificar a busca aleatória como referência
básica e apresentar o TPE como alternativa prática para espaços condicionais.
Isso não significa que o TPE sempre vencerá. Com poucas avaliações, muito ruído
ou um espaço pequeno, o modelo substituto pode não oferecer vantagem sobre uma
busca aleatória bem definida.

---

## 2. Auto-PyTorch

### Problema e proposta

O **Auto-PyTorch Tabular** automatiza aprendizado profundo em dados tabulares.
Sua proposta é otimizar conjuntamente a arquitetura da rede e os
hiperparâmetros de treinamento, pois a melhor taxa de aprendizado,
regularização ou técnica de treinamento pode depender da arquitetura escolhida.
O sistema também automatiza pré-processamento, seleção de modelos e formação de
*ensembles*.

Ele combina quatro elementos:

1. **Espaço de busca conjunto**, com decisões de arquitetura e treinamento.
2. **BOHB e múltiplas fidelidades**, avaliando muitas configurações com poucos
   recursos e aumentando o orçamento das mais promissoras.
3. **Portfólio para warm start**, que transfere configurações encontradas em
   tarefas anteriores.
4. **Seleção de ensemble**, combinando redes e modelos tradicionais.

O objetivo é alcançar bom desempenho *anytime*: produzir soluções úteis cedo e
continuar melhorando enquanto houver orçamento.

### Espaço de busca

O artigo estuda dois espaços. O primeiro possui sete hiperparâmetros e usa MLPs
em formato de funil, descritas pelo número de camadas e pelo número máximo de
unidades. O espaço completo inclui MLPs e ResNets, SGD e Adam, *mixup*,
Shake-Shake, ShakeDrop e pré-processamento com SVD truncada.

As redes são parametrizadas por formatos e blocos repetidos em vez de cada
camada ser configurada isoladamente. Isso reduz a dimensionalidade sem eliminar
decisões importantes. O espaço é hierárquico, pois certas escolhas ativam ou
desativam outros hiperparâmetros.

### BOHB e múltiplas fidelidades

O **BOHB** combina otimização bayesiana com Hyperband. O orçamento corresponde
ao número de épocas, usando 12, 25 e 50 épocas no estudo principal. O
*Successive Halving* avalia muitas configurações no menor orçamento e promove
as mais promissoras. Um modelo probabilístico concentra novas amostras em
regiões favoráveis, mantendo uma parcela de exploração aleatória. A arquitetura
mestre-trabalhador permite avaliações paralelas.

Esse ganho depende de as fidelidades menores informarem o resultado final. Se a
ordenação das configurações mudar muito entre poucas e muitas épocas, candidatos
bons no longo prazo podem ser eliminados cedo.

### LCBench

Para estudar essa hipótese, os autores introduzem o **LCBench**, um *benchmark*
com curvas de 2.000 configurações em 35 conjuntos tabulares, três sementes e
três orçamentos. A coleta consumiu aproximadamente 1.500 horas de CPU.

Entre os resultados:

- os melhores resultados dos 35 conjuntos envolvem 22 configurações distintas,
  e uma configuração é a melhor em até sete tarefas;
- um portfólio de dez configurações alcança arrependimento médio de acurácia
  inferior a 0,1% em relação aos 2.000 candidatos;
- no espaço reduzido, as correlações médias de postos de Spearman entre
  orçamentos variam de 0,88 a 0,96;
- a importância dos hiperparâmetros é relativamente estável entre orçamentos;
- a fANOVA global destaca o número de camadas, enquanto a análise local destaca
  a taxa de aprendizado.

Os resultados apoiam múltiplas fidelidades e transferência de configurações,
embora a correlação entre orçamentos seja menor em algumas tarefas do espaço
completo.

### Meta-learning e ensembles

O portfólio é construído a partir de 100 conjuntos de meta-treinamento. As
configurações incumbentes são avaliadas nas demais tarefas, e uma seleção gulosa
escolhe 16 configurações complementares que minimizam o arrependimento médio.
Em uma nova tarefa, o BOHB começa pelo portfólio e depois segue sua amostragem
convencional. O método dispensa meta-atributos, mas depende de as tarefas
anteriores representarem as futuras.

As predições produzidas durante a busca são reutilizadas em uma seleção gulosa
de *ensemble*. O conjunto inclui redes e Random Forest, Extra Trees, LightGBM,
CatBoost e k-NN. Essa diversidade é importante porque redes profundas não são
sempre a melhor escolha para dados tabulares.

### Resultados e limitações

Nas ablações com oito conjuntos de meta-teste, o BOHB apresenta desempenho
*anytime* igual ou melhor que a otimização bayesiana convencional, sobretudo
nas tarefas maiores. O portfólio melhora a fase inicial, e o *ensemble* aumenta
a robustez. Três trabalhadores alcançam aceleração de três vezes ou mais nas
tarefas maiores em relação à execução sequencial.

Após uma hora, o Auto-PyTorch obtém o melhor resultado geral contra Auto-Net
2.0, AutoKeras, auto-sklearn e hyperopt-sklearn. Em *covertype*, o erro cai de
31,78% com Auto-Net 2.0 para 3,14%. Contra o AutoGluon, o quadro é mais
equilibrado: sem combinação de modelos, o Auto-PyTorch apresenta melhor
resultado geral; com *ensemble* e *stacking*, o AutoGluon obtém pequena
vantagem. Não há, portanto, superioridade universal.

No NAS-Bench-201, o núcleo de BOHB e portfólio supera os métodos *one-shot*
comparados, exceto GDAS, em desempenho *anytime*. Esse teste é uma prova de
conceito, não uma validação abrangente em imagens.

As principais limitações são:

- o LCBench exclui os quatro maiores conjuntos, usa um espaço reduzido e apenas
  três fidelidades;
- a avaliação principal contém somente oito conjuntos de meta-teste e usa
  acurácia, limitada em problemas desbalanceados;
- o meta-learning depende da representatividade das tarefas anteriores;
- o *ensemble* pode sobreajustar a validação reutilizada na busca;
- os modelos tradicionais usam hiperparâmetros padrão;
- a generalização para outras modalidades recebe evidência restrita.

---

## 3. Optuna

### Conceito e funcionamento

O **Optuna** é uma biblioteca geral para otimização de hiperparâmetros. Não é
um sistema AutoDL completo nem uma contribuição experimental equivalente aos
dois artigos. Seu papel é fornecer uma interface para definir espaços de busca,
executar avaliações, registrar resultados, selecionar amostradores e
interromper tentativas pouco promissoras.

Um `Study` representa o processo de otimização, e cada execução da função
objetivo é um `Trial`. Métodos como `suggest_float`, `suggest_int` e
`suggest_categorical` definem os valores testados. Espaços condicionais podem
ser expressos diretamente no fluxo Python da função objetivo.

### Samplers e pruning

O *sampler* escolhe a próxima configuração. O `RandomSampler` implementa busca
aleatória e serve como baseline. O `TPESampler` aplica a família de métodos TPE
apresentada por Bergstra et al. Outros amostradores atendem cenários como
otimização multiobjetivo.

Um *pruner* encerra um `Trial` quando métricas intermediárias indicam baixo
potencial. O princípio é semelhante ao de múltiplas fidelidades: não gastar o
orçamento máximo em candidatos ruins. Contudo, combinar um *sampler* e um
*pruner* do Optuna não reproduz automaticamente o BOHB do Auto-PyTorch. O
comportamento depende da estratégia, da métrica intermediária e da função
objetivo implementada.

O Optuna também oferece persistência, paralelismo, retomada de estudos,
visualizações, integrações e otimização multiobjetivo. Ele organiza a busca,
mas não corrige uma função objetivo mal definida. Separação de dados,
distribuições dos parâmetros, sementes, orçamento e métricas continuam sendo
responsabilidade do experimento.

### Relação com a reprodução do seminário

O notebook do seminário usa o Optuna didaticamente para comparar busca
aleatória e TPE, registrar os *trials* e recuperar a melhor configuração. Ele
não reproduz o Auto-PyTorch, pois não inclui BOHB, LCBench, portfólio
meta-aprendido ou seleção de *ensemble*.

---

## 4. Comparação dos três

| Aspecto | Bergstra et al. (2011) | Auto-PyTorch (2021) | Optuna |
| --- | --- | --- | --- |
| Natureza | artigo sobre algoritmos de HPO | artigo e sistema AutoDL | biblioteca de HPO |
| Problema | escolher hiperparâmetros | automatizar arquitetura, treinamento e modelos | executar e gerenciar estudos |
| Unidade avaliada | configuração de hiperparâmetros | pipeline, arquitetura e treinamento | função objetivo do usuário |
| Estratégias | busca aleatória, GP e TPE | BOHB, portfólio e *ensemble* | *samplers*, *pruners* e paralelismo |
| Histórico | TPE e GP usam avaliações anteriores | BOHB usa avaliações; portfólio usa tarefas anteriores | depende do *sampler* |
| Fidelidades | não são o foco | épocas: 12, 25 e 50 | métricas intermediárias e *pruning* |
| Meta-learning | não | portfólio sem meta-atributos | não no fluxo básico |
| Ensemble | não | redes e modelos tradicionais | fora do foco da biblioteca |
| Espaços condicionais | estrutura em árvore do TPE | arquitetura e treinamento hierárquicos | lógica da função objetivo |
| Evidência | redes e *deep belief networks* | LCBench, meta-testes e NAS-Bench-201 | depende da aplicação |
| Vantagem | fundamenta busca aleatória e TPE | solução AutoDL integrada | aplicação flexível e prática |
| Limitação | não é pipeline AutoML completo | custo e complexidade maiores | não define sozinho um experimento correto |

### Relação conceitual

Bergstra et al. fornecem a base algorítmica: explicam por que a busca aleatória
é um baseline importante e como o TPE aprende com o histórico. O Optuna torna
essas ideias acessíveis em software, permitindo alternar entre
`RandomSampler` e `TPESampler`, registrar avaliações e adicionar *pruning*.

O Auto-PyTorch atua em um nível mais amplo. Ele não usa Optuna nos experimentos
do artigo, mas BOHB, que combina otimização bayesiana e Hyperband. Também
incorpora transferência entre tarefas, arquiteturas específicas,
pré-processamento, paralelismo e *ensembles*.

Em síntese:

$$
\text{Bergstra et al.} \rightarrow \text{fundamentos de HPO e TPE},
$$

$$
\text{Optuna} \rightarrow \text{infraestrutura prática para executar HPO},
$$

$$
\text{Auto-PyTorch} \rightarrow \text{sistema AutoDL especializado e integrado}.
$$

### Quando usar

- **Busca aleatória:** primeiro baseline para medir se uma estratégia adaptativa
  realmente ajuda sob o mesmo orçamento.
- **TPE com Optuna:** quando avaliações são caras, já existe histórico útil e o
  espaço possui parâmetros contínuos, categóricos ou condicionais.
- **Pruning com Optuna:** quando o treinamento fornece métricas intermediárias
  informativas e pode ser interrompido com segurança.
- **Auto-PyTorch:** quando se deseja automatizar uma pipeline tabular completa e
  há orçamento para a busca e o *ensemble*.

## Conclusão geral

Os três representam uma progressão de escopo. Bergstra et al. estabelecem como
comparar e selecionar configurações melhor do que uma grade. O Optuna oferece
abstrações para aplicar essas estratégias. O Auto-PyTorch amplia o problema
para AutoDL, coordenando arquitetura, treinamento, fidelidade, transferência e
combinação de modelos.

A mensagem comum é que HPO não elimina a necessidade de um bom projeto
experimental. O desempenho continua condicionado pelo espaço de busca, pela
métrica, pela separação dos dados, pelo orçamento e pela relação entre
avaliações parciais e finais. Busca aleatória, TPE, *pruning* e BOHB distribuem
melhor esse orçamento, mas não garantem superioridade isoladamente.

## Referências

1. Bergstra, J. S.; Bardenet, R.; Bengio, Y.; Kégl, B. *Algorithms for
   Hyper-Parameter Optimization*. Advances in Neural Information Processing
   Systems 24, 2011.
2. Zimmer, L.; Lindauer, M.; Hutter, F. *Auto-PyTorch: Multi-Fidelity
   MetaLearning for Efficient and Robust AutoDL*. IEEE Transactions on Pattern
   Analysis and Machine Intelligence, 2021.
3. [Optuna Tutorial 3.4.1](https://optuna.readthedocs.io/en/v3.4.1/tutorial/index.html).
