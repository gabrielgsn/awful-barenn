# Barenn: resumo da entrega

Fork de `emil-freme/awful-barenn`. A rede é uma MLP 64-32-16-10 (2.778 pesos) que classifica os dígitos do `optdigits` em 51 épocas. Cada otimização tem uma issue com diagnóstico, hipótese e métrica base, uma branch `issue/NN`, uma tag anotada e um comentário com o resultado. O enunciado original está depois deste resumo.

## Resultados

Todas as versões foram medidas na mesma sessão, no container Ubuntu 22.04 de `docker/Dockerfile`: i7-11800H (8 núcleos, 16 threads), 15 GB de RAM, RTX 3050 Laptop (16 SMs, driver 610.60), g++ 11.4 com `-O3` e `nvcc` 12.6 com `-arch=sm_86`. O tempo é só o do treino: média das medianas de 2 rodadas alternadas, com 15 execuções por rodada. Os dados completos estão na [issue #12](https://github.com/gabrielgsn/awful-barenn/issues/12).

| Versão | Tag | Tempo de treino | Speedup sobre o legado | Acurácia |
|:--|:--|--:|--:|--:|
| Baseline Legado | [`v0.0-baseline`](https://github.com/gabrielgsn/awful-barenn/tree/v0.0-baseline) | 1,249 s | 1,0x | 95,5% |
| CPU Otimizada, sequencial | [`v0.5-sem-grad-in`](https://github.com/gabrielgsn/awful-barenn/tree/v0.5-sem-grad-in) | 0,335 s | 3,7x | 95,5% |
| CPU Otimizada, lote 32 e OpenMP com 4 threads | [`v0.7-openmp`](https://github.com/gabrielgsn/awful-barenn/tree/v0.7-openmp) | 0,184 s | 6,8x | 96,4% |
| CUDA Inicial | [`v1.0-cuda-inicial`](https://github.com/gabrielgsn/awful-barenn/tree/v1.0-cuda-inicial) | 0,532 s | 2,3x | 96,4% |
| CUDA Otimizado, kernel persistente | [`v1.2-cuda-avancado`](https://github.com/gabrielgsn/awful-barenn/tree/v1.2-cuda-avancado) | 0,1035 s | 12,1x | 96,4% |

Observações:

- A partir do lote 32 o algoritmo muda: os pesos são atualizados a cada 32 amostras, então o resultado final não é o do legado. Com lote 1 a saída é idêntica à do legado. Por isso a versão sequencial (3,7x, mesma saída) aparece separada.
- A primeira versão CUDA é 2,9x mais lenta que a CPU com 4 threads, porque lança 67.320 kernels pequenos. A versão final é 1,8x mais rápida que essa CPU e 5,1x mais rápida que a CUDA inicial.
- O tempo varia entre rodadas (o legado deu 1,36 s numa e 1,13 s na outra), então os speedups têm alguns por cento de incerteza.
- A GPU usa só 1 dos 16 SMs (Nsight Compute: 6,25% dos ciclos de SM da GPU e ocupação de 66,7% nesse SM). A versão CUDA aceita lote de até 32, e com a taxa atual o treino diverge com lote 64 (issue #7).

## Issues e tags

| Issue | O que mudou | Tag | Resultado |
|:-:|:--|:--|:--|
| [#1](https://github.com/gabrielgsn/awful-barenn/issues/1) | Baseline e profiling com callgrind | [`v0.0-baseline`](https://github.com/gabrielgsn/awful-barenn/tree/v0.0-baseline) | `Backward` 67,7% e `Forward` 23,8% das instruções |
| [#2](https://github.com/gabrielgsn/awful-barenn/issues/2) | Compilar com `-O3` | [`v0.1-flags-O3`](https://github.com/gabrielgsn/awful-barenn/tree/v0.1-flags-O3) | 1,5x; `-march=native` descartado (muda a saída e fica mais lento) |
| [#3](https://github.com/gabrielgsn/awful-barenn/issues/3) | `Backward` percorrendo os pesos por linha, junto com a atualização | [`v0.2-backward-por-linha`](https://github.com/gabrielgsn/awful-barenn/tree/v0.2-backward-por-linha) | 1,6x sobre `v0.1` |
| [#4](https://github.com/gabrielgsn/awful-barenn/issues/4) | `Forward` com 4 neurônios por vez | [`v0.3-forward-ilp`](https://github.com/gabrielgsn/awful-barenn/tree/v0.3-forward-ilp) | 1,3x |
| [#5](https://github.com/gabrielgsn/awful-barenn/issues/5) | Vetores por referência, sem cópias | [`v0.4-sem-copias`](https://github.com/gabrielgsn/awful-barenn/tree/v0.4-sem-copias) | ganho pequeno, com menos instruções e `malloc` |
| [#6](https://github.com/gabrielgsn/awful-barenn/issues/6) | Sem `grad_in` na primeira camada | [`v0.5-sem-grad-in`](https://github.com/gabrielgsn/awful-barenn/tree/v0.5-sem-grad-in) | 3,7x sobre o legado no total |
| [#7](https://github.com/gabrielgsn/awful-barenn/issues/7) | Mini-batch serial | [`v0.6-minibatch`](https://github.com/gabrielgsn/awful-barenn/tree/v0.6-minibatch) | lote 32 com 96,4%; lote 64 diverge |
| [#8](https://github.com/gabrielgsn/awful-barenn/issues/8) | OpenMP nos blocos do lote | [`v0.7-openmp`](https://github.com/gabrielgsn/awful-barenn/tree/v0.7-openmp) | 1,7x a 1,9x com 4 threads; mesma saída de 1 a 16 threads |
| [#9](https://github.com/gabrielgsn/awful-barenn/issues/9) | Porte para CUDA | [`v1.0-cuda-inicial`](https://github.com/gabrielgsn/awful-barenn/tree/v1.0-cuda-inicial) | mais lenta que a CPU por causa dos lançamentos de kernel |
| [#10](https://github.com/gabrielgsn/awful-barenn/issues/10) | Contexto CUDA fora do cronômetro e um kernel por lote em memória compartilhada | [`v1.1-cuda-otimizado`](https://github.com/gabrielgsn/awful-barenn/tree/v1.1-cuda-otimizado) | 4,1x sobre `v1.0` |
| [#11](https://github.com/gabrielgsn/awful-barenn/issues/11) | Kernel persistente com prefetch | [`v1.2-cuda-avancado`](https://github.com/gabrielgsn/awful-barenn/tree/v1.2-cuda-avancado) | 1,26x sobre `v1.1` |
| [#12](https://github.com/gabrielgsn/awful-barenn/issues/12) | Este resumo e a medição final | [`v2.0-entrega`](https://github.com/gabrielgsn/awful-barenn/tree/v2.0-entrega) | todas as versões na mesma sessão |

## Como reproduzir

```bash
git checkout <tag>
make check          # CPU: compara com golden.txt (lote 1) e golden_lote32.txt (lote 32)
make check-cuda     # CUDA (precisa de nvcc e GPU sm_86): compara com tolerância
./bench.sh 15 ./barenn optdigits.tra optdigits.tes 32

docker build -t barenn-env docker
docker run --rm --gpus all -v "$PWD":/work -w /work barenn-env make check check-cuda
```

No Windows, com `nvcc` 13.3, a versão CUDA dá 1732 acertos e perda 0,0110. No container dá 1733 e 0,0108. Os dois passam em `make check-cuda`.

---
# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
